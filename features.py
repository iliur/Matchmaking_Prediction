"""
Feature helpers for matchmaking:
- Resolves Riot IDs to PUUIDs (when needed)
- Pulls Match-V5 data (SoloQ by default) with pagination(100 max per call)
- Caches match JSONs to disk to avoid refetch on reruns
- Builds *pre-match* rolling stats per player (no leakage)
- Aggregates to team features and team deltas
- Runtime: supports computing team deltas from Riot IDs **or** directly from PUUIDs
"""

from __future__ import annotations

import os, json
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from riotwatcher import LolWatcher, RiotWatcher

# ---------- config & clients ----------

load_dotenv(find_dotenv())

API_KEY  = os.getenv("RIOT_API_KEY")
PLATFORM = os.getenv("RIOT_PLATFORM", "na1")
REGION   = os.getenv("RIOT_REGION", "americas")
CACHE_DIR = os.getenv("RIOT_CACHE_DIR", ".riot_cache")

if not API_KEY:
    raise RuntimeError("Missing RIOT_API_KEY in .env or environment")

lol  = LolWatcher(API_KEY)
riot = RiotWatcher(API_KEY)

# SoloQ by default
QUEUE_SOLO = 420

# The feature names we produce for training and runtime scoring
FEATURE_NAMES: List[str] = [
    # team A aggregates (handy for inspection)
    "A_rank_mean", "A_wr_mean", "A_kda_mean", "A_csmin_mean", "A_dpm_mean", "A_kp_mean", "A_role_rate_mean",
    # team deltas (Team A - Team B) — what we actually train/score on
    "delta_rank_mean", "delta_wr_mean", "delta_kda_mean", "delta_csmin_mean", "delta_dpm_mean", "delta_kp_mean", "delta_role_rate_mean",
]

# ---------- small utilities ----------

def tier_to_score(tier: str, rank: str, lp: int) -> float:
    base = {
        "IRON":1, "BRONZE":2, "SILVER":3, "GOLD":4, "PLATINUM":5,
        "EMERALD":6, "DIAMOND":7, "MASTER":8, "GRANDMASTER":8.5, "CHALLENGER":9
    }.get((tier or "").upper(), 0)
    sub  = {"IV":0.0, "III":0.25, "II":0.5, "I":0.75}.get((rank or "").upper(), 0.0)
    return base + sub + (lp or 0)/100.0

def rank_score_by_puuid(puuid: str) -> float:
    """Convert current ranked (if any) to a numeric score; 0 if unranked."""
    try:
        summ = lol.summoner.by_puuid(PLATFORM, puuid)
        entries = lol.league.by_summoner(PLATFORM, summ["id"])
        solo = next((e for e in entries if e.get("queueType") == "RANKED_SOLO_5x5"), None)
        e = solo or (entries[0] if entries else None)
        if not e:
            return 0.0
        return tier_to_score(e.get("tier"), e.get("rank"), e.get("leaguePoints", 0))
    except Exception:
        return 0.0

def riot_id_to_puuid(riot_id: str) -> str:
    """Riot ID 'Game#Tag' -> PUUID via Account-V1."""
    game, tag = riot_id.split("#", 1)
    acc = riot.account.by_riot_id(REGION, game, tag)
    return acc["puuid"]

def puuid_to_riot_id(puuid: str) -> str:
    """PUUID -> Riot ID via Account-V1 (safe even if the player renamed)."""
    acc = riot.account.by_puuid(REGION, puuid)
    return f"{acc.get('gameName')}#{acc.get('tagLine')}"

# ---------- match list pagination & caching ----------

def _match_ids_paginated(puuid: str, total: int) -> List[str]:
    """Respect Match-V5 limit: max 100 per call. Paginate until `total` or no more results."""
    ids: List[str] = []
    start = 0
    remaining = max(0, int(total))
    while remaining > 0:
        batch_size = min(remaining, 100)
        batch = lol.match.matchlist_by_puuid(REGION, puuid, start=start, count=batch_size)
        if not batch:
            break
        ids.extend(batch)
        if len(batch) < batch_size:  # server ran out
            break
        start += len(batch)
        remaining -= batch_size
    return ids

def _cache_path(match_id: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{match_id}.json")

def _load_match_from_cache(match_id: str) -> Optional[dict]:
    path = _cache_path(match_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _save_match_to_cache(match_id: str, data: dict) -> None:
    try:
        with open(_cache_path(match_id), "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def _get_match(match_id: str) -> dict:
    cached = _load_match_from_cache(match_id)
    if cached is not None:
        return cached
    m = lol.match.by_id(REGION, match_id)
    _save_match_to_cache(match_id, m)
    return m

def fetch_recent_matches_for_puuid(puuid: str, count: int = 120, queue_id: Optional[int] = QUEUE_SOLO) -> List[dict]:
    """
    Grab recent matches; filter by queue if provided.
    Uses pagination so any count works (in chunks of 100). Caches matches to disk.
    """
    match_ids = _match_ids_paginated(puuid, total=count)
    out = []
    for i, mid in enumerate(match_ids, 1):
        if i % 20 == 1:
            print(f"[fetch] {puuid[:8]}… match {i}/{len(match_ids)}")
        m = _get_match(mid)
        if queue_id is not None and m.get("info", {}).get("queueId") != queue_id:
            continue
        out.append(m)
    return out

# ---------- pre-match rolling stats (no leakage) ----------

def _player_stats_from_participant(info: dict, p: dict) -> Dict[str, float]:
    dur_min = max(1, info.get("gameDuration", 1)) / 60.0
    kills = p.get("kills", 0)
    deaths = p.get("deaths", 0)
    assists = p.get("assists", 0)
    cs = p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)
    chal = p.get("challenges", {}) or {}
    dpm = chal.get("damagePerMinute")
    kp = chal.get("killParticipation")
    if dpm is None:
        dpm = (p.get("totalDamageDealtToChampions", 0) / max(1, info.get("gameDuration", 1))) * 60.0
    if kp is None:
        # crude fallback using team kills
        team_kills = 1
        for t in info.get("teams", []):
            if t.get("teamId") == p.get("teamId"):
                team_kills = max(1, t.get("objectives", {}).get("champion", {}).get("kills", 1))
        kp = (kills + assists) / team_kills
        kp = max(0.0, min(1.0, kp))
    return dict(
        win = 1.0 if p.get("win") else 0.0,
        kda = (kills + assists) / max(1.0, deaths),
        csmin = cs / dur_min,
        dpm = dpm,
        kp = kp,
        role = p.get("teamPosition") or p.get("role") or p.get("lane") or "UNK",
    )

def build_player_pre_match_stats(puuid: str, matches_for_player: List[dict], cutoff_ms: int, window: int = 30) -> Dict[str, float]:
    """
    Compute rolling stats for a single player using only matches with gameStartTimestamp < cutoff_ms.
    Returns defaults if no history.
    """
    history = []
    for m in matches_for_player:
        info = m["info"]
        if info.get("gameStartTimestamp", 0) >= cutoff_ms:
            continue
        part = next((pp for pp in info.get("participants", []) if pp.get("puuid") == puuid), None)
        if not part:
            continue
        history.append(_player_stats_from_participant(info, part))

    history = history[:window]
    if not history:
        return dict(wr=0.5, kda=2.0, csmin=5.0, dpm=300.0, kp=0.5, role_rate=0.5, games=0)

    df = pd.DataFrame(history)
    wr = df["win"].mean()
    kda = df["kda"].mean()
    csmin = df["csmin"].mean()
    dpm = df["dpm"].mean()
    kp = df["kp"].mean()
    role_rate = df["role"].value_counts(normalize=True).max()
    return dict(wr=wr, kda=kda, csmin=csmin, dpm=dpm, kp=kp, role_rate=role_rate, games=len(df))

# ---------- dataset builder (training) ----------

def build_dataset_for_riot_ids(
    riot_ids: List[str],
    per_player_matches: int = 120,
    queue_id: Optional[int] = QUEUE_SOLO,
    window: int = 30,
) -> pd.DataFrame:
    """
    Build a team-level dataset with two rows per match (team100 and team200).
    Each row includes Team A aggregates + Team A vs Team B deltas, and label 'win'.
    """
    print(f"[dataset] seeds: {riot_ids}")
    # 1) resolve all PUUIDs
    puuids = {rid: riot_id_to_puuid(rid) for rid in riot_ids}
    print(f"[dataset] resolved {len(puuids)} PUUIDs")

    # 2) fetch matches for each player and collect a unique set
    matches_by_id: Dict[str, dict] = {}
    per_player_matches_map: Dict[str, List[dict]] = {}
    for rid, puuid in puuids.items():
        print(f"[dataset] pulling recent matches for {rid} (puuid {puuid[:8]}…) count={per_player_matches}")
        ms = fetch_recent_matches_for_puuid(puuid, count=per_player_matches, queue_id=queue_id)
        per_player_matches_map[puuid] = ms
        for m in ms:
            matches_by_id[m["metadata"]["matchId"]] = m

    matches = list(matches_by_id.values())
    print(f"[dataset] unique matches collected: {len(matches)}")

    rows: List[Dict[str, float]] = []

    for idx, m in enumerate(matches, 1):
        if idx % 20 == 1:
            print(f"[dataset] processing match {idx}/{len(matches)}")
        info = m["info"]
        if queue_id is not None and info.get("queueId") != queue_id:
            continue

        team100 = [p for p in info["participants"] if p.get("teamId") == 100]
        team200 = [p for p in info["participants"] if p.get("teamId") == 200]
        if len(team100) != 5 or len(team200) != 5:
            continue

        cutoff = info.get("gameStartTimestamp", 0)

        def team_agg(team_parts: List[dict]) -> Dict[str, float]:
            feats = []
            for p in team_parts:
                puuid = p["puuid"]
                pre = build_player_pre_match_stats(
                    puuid,
                    per_player_matches_map.get(puuid, []),
                    cutoff_ms=cutoff,
                    window=window,
                )
                rs = rank_score_by_puuid(puuid)
                feats.append(dict(rank=rs, **pre))
            df = pd.DataFrame(feats)
            return {
                "rank_mean": df["rank"].mean(),
                "wr_mean": df["wr"].mean(),
                "kda_mean": df["kda"].mean(),
                "csmin_mean": df["csmin"].mean(),
                "dpm_mean": df["dpm"].mean(),
                "kp_mean": df["kp"].mean(),
                "role_rate_mean": df["role_rate"].mean(),
            }

        A = team_agg(team100)
        B = team_agg(team200)

        row_A = {
            "A_rank_mean": A["rank_mean"], "A_wr_mean": A["wr_mean"], "A_kda_mean": A["kda_mean"],
            "A_csmin_mean": A["csmin_mean"], "A_dpm_mean": A["dpm_mean"], "A_kp_mean": A["kp_mean"],
            "A_role_rate_mean": A["role_rate_mean"],

            "delta_rank_mean": A["rank_mean"] - B["rank_mean"],
            "delta_wr_mean": A["wr_mean"] - B["wr_mean"],
            "delta_kda_mean": A["kda_mean"] - B["kda_mean"],
            "delta_csmin_mean": A["csmin_mean"] - B["csmin_mean"],
            "delta_dpm_mean": A["dpm_mean"] - B["dpm_mean"],
            "delta_kp_mean": A["kp_mean"] - B["kp_mean"],
            "delta_role_rate_mean": A["role_rate_mean"] - B["role_rate_mean"],

            "win": 1 if team100[0].get("win") else 0,
            "match_id": m["metadata"]["matchId"],
            "start_ms": info.get("gameStartTimestamp"),
            "queueId": info.get("queueId"),
            "version": info.get("gameVersion"),
            "team": 100,
        }
        row_B = {
            "A_rank_mean": B["rank_mean"], "A_wr_mean": B["wr_mean"], "A_kda_mean": B["kda_mean"],
            "A_csmin_mean": B["csmin_mean"], "A_dpm_mean": B["dpm_mean"], "A_kp_mean": B["kp_mean"],
            "A_role_rate_mean": B["role_rate_mean"],

            "delta_rank_mean": B["rank_mean"] - A["rank_mean"],
            "delta_wr_mean": B["wr_mean"] - A["wr_mean"],
            "delta_kda_mean": B["kda_mean"] - A["kda_mean"],
            "delta_csmin_mean": B["csmin_mean"] - A["csmin_mean"],
            "delta_dpm_mean": B["dpm_mean"] - A["dpm_mean"],
            "delta_kp_mean": B["kp_mean"] - A["kp_mean"],
            "delta_role_rate_mean": B["role_rate_mean"] - A["role_rate_mean"],

            "win": 1 if team200[0].get("win") else 0,
            "match_id": m["metadata"]["matchId"],
            "start_ms": info.get("gameStartTimestamp"),
            "queueId": info.get("queueId"),
            "version": info.get("gameVersion"),
            "team": 200,
        }
        rows.extend([row_A, row_B])

    df = pd.DataFrame(rows)
    cols = FEATURE_NAMES + ["win", "match_id", "start_ms", "queueId", "version", "team"]
    print(f"[dataset] rows built: {len(df)}")
    return df[cols]

# ---------- runtime features ----------

def _team_aggregate_from_puuids(puuids: List[str], history_count: int, window: int) -> Dict[str, float]:
    feats = []
    for puuid in puuids:
        ms = fetch_recent_matches_for_puuid(puuid, count=history_count, queue_id=QUEUE_SOLO)
        pre = build_player_pre_match_stats(puuid, ms, cutoff_ms=10**18, window=window)  # 'now' cutoff
        rs = rank_score_by_puuid(puuid)
        feats.append(dict(rank=rs, **pre))
    df = pd.DataFrame(feats)
    return {
        "rank_mean": df["rank"].mean(),
        "wr_mean": df["wr"].mean(),
        "kda_mean": df["kda"].mean(),
        "csmin_mean": df["csmin"].mean(),
        "dpm_mean": df["dpm"].mean(),
        "kp_mean": df["kp"].mean(),
        "role_rate_mean": df["role_rate"].mean(),
    }

def runtime_team_deltas_from_puuids(teamA_puuids: List[str], teamB_puuids: List[str],
                                    window: int = 20, history_count: int = 60) -> Dict[str, float]:
    """Runtime features using PUUIDs only (no Riot-ID lookups)."""
    if len(teamA_puuids) != 5 or len(teamB_puuids) != 5:
        raise ValueError("Provide exactly 5 PUUIDs per team.")
    A = _team_aggregate_from_puuids(teamA_puuids, history_count, window)
    B = _team_aggregate_from_puuids(teamB_puuids, history_count, window)
    return {
        "A_rank_mean": A["rank_mean"], "A_wr_mean": A["wr_mean"], "A_kda_mean": A["kda_mean"],
        "A_csmin_mean": A["csmin_mean"], "A_dpm_mean": A["dpm_mean"], "A_kp_mean": A["kp_mean"],
        "A_role_rate_mean": A["role_rate_mean"],
        "delta_rank_mean": A["rank_mean"] - B["rank_mean"],
        "delta_wr_mean": A["wr_mean"] - B["wr_mean"],
        "delta_kda_mean": A["kda_mean"] - B["kda_mean"],
        "delta_csmin_mean": A["csmin_mean"] - B["csmin_mean"],
        "delta_dpm_mean": A["dpm_mean"] - B["dpm_mean"],
        "delta_kp_mean": A["kp_mean"] - B["kp_mean"],
        "delta_role_rate_mean": A["role_rate_mean"] - B["role_rate_mean"],
    }

def runtime_team_deltas(teamA_riot_ids: List[str], teamB_riot_ids: List[str],
                        window: int = 20, history_count: int = 60) -> Dict[str, float]:
    """Runtime features using Riot IDs (kept for POST /pwin)."""
    if len(teamA_riot_ids) != 5 or len(teamB_riot_ids) != 5:
        raise ValueError("Provide exactly 5 Riot IDs per team (Name#TAG).")
    teamA_puuids = [riot_id_to_puuid(rid) for rid in teamA_riot_ids]
    teamB_puuids = [riot_id_to_puuid(rid) for rid in teamB_riot_ids]
    return runtime_team_deltas_from_puuids(teamA_puuids, teamB_puuids, window=window, history_count=history_count)
