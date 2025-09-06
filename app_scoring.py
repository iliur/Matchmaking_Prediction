"""
FastAPI scoring service:
- POST /pwin -> given two teams (5 Riot IDs each), returns P(Team A wins)
- GET  /pwin_from_match/{match_id} -> builds teams from a cached match (PUUIDs) and scores (Team 100 as A)
- GET  /match_info/{match_id} -> human-friendly summary + external links
- GET  /recent_matches/riotid/{game}/{tag}?count=10&queue=420 -> list recent matches (fresh ids + links)
"""
# queue=420 is soloq

from __future__ import annotations

import os, sys, re, datetime
import joblib
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from features import (
    FEATURE_NAMES,
    _get_match,
    runtime_team_deltas,
    runtime_team_deltas_from_puuids,
    puuid_to_riot_id,
    riot_id_to_puuid,
    fetch_recent_matches_for_puuid,
)

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

app = FastAPI(title="Matchmaking P(win) Scoring", version="0.1.6")

# ---- helpers ----
def _log_link(match_id: str, region_code: str = "na") -> str:
    """
    LeagueOfGraphs link helper.
    Accepts 'NA1_5255517869' or 'NA1-5255517869' or just '5255517869'
    and returns 'https://www.leagueofgraphs.com/match/na/5255517869'
    """
    m = re.search(r"(\d+)$", match_id)
    numeric = m.group(1) if m else match_id.replace("_", "-")
    return f"https://www.leagueofgraphs.com/match/{region_code}/{numeric}"

# ----- startup: load model -----
@app.on_event("startup")
def _startup():
    print("[startup] python:", sys.executable)
    try:
        import sklearn
        print("[startup] sklearn:", sklearn.__version__)
    except Exception:
        print("[startup] sklearn: <unavailable>")
    print("[startup] joblib:", joblib.__version__)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Train one with train_baseline.py")

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Failed to load model.pkl — run uvicorn from the SAME env used to train.\n"
            "Example: `conda activate venv && python -m uvicorn app_scoring:app --reload`\n"
            f"Original error: {type(e).__name__}: {e}"
        )
    app.state.model = bundle["model"]
    app.state.model_features = bundle["feature_names"]
    print("[startup] model loaded.")

# ----- models -----
class TeamsRequest(BaseModel):
    teamA: List[str]
    teamB: List[str]

    @field_validator("teamA")
    @classmethod
    def _five_a(cls, v):
        if len(v) != 5:
            raise ValueError("teamA must have exactly 5 Riot IDs (Name#TAG)")
        return v

    @field_validator("teamB")
    @classmethod
    def _five_b(cls, v):
        if len(v) != 5:
            raise ValueError("teamB must have exactly 5 Riot IDs (Name#TAG)")
        return v

# ----- endpoints -----
@app.get("/health")
def health():
    feats = getattr(app.state, "model_features", FEATURE_NAMES)
    return {"status": "ok", "loaded_features": feats}

@app.post("/pwin")
def pwin(req: TeamsRequest):
    """Score using Riot IDs (kept for general use)."""
    try:
        feats: Dict[str, float] = runtime_team_deltas(req.teamA, req.teamB, window=20, history_count=60)
    except Exception as e:
        raise HTTPException(400, f"feature computation failed: {e}")

    model = getattr(app.state, "model", None)
    model_features = getattr(app.state, "model_features", None)
    if model is None or model_features is None:
        raise HTTPException(500, "Model not loaded on startup.")

    X = [[feats[name] for name in model_features]]
    prob = float(model.predict_proba(X)[:, 1])
    return {
        "p_win_A": round(prob, 4),
        "features_used": model_features,
        "values": {k: round(feats[k], 4) for k in model_features},
    }

@app.get("/pwin_from_match/{match_id}")
def pwin_from_match(match_id: str, history_count: int = 60, window: int = 20):
    """
    Robust scoring from a cached match id (PUUID-only, no Riot-ID lookups):
    - Team 100 -> A, Team 200 -> B
    """
    try:
        m = _get_match(match_id)
    except Exception as e:
        raise HTTPException(404, f"Could not load match {match_id}: {e}")

    info = m.get("info", {})
    parts = info.get("participants", [])
    if len(parts) != 10:
        raise HTTPException(400, "Match does not have 10 participants.")

    team100_puuids = [p["puuid"] for p in parts if p.get("teamId") == 100]
    team200_puuids = [p["puuid"] for p in parts if p.get("teamId") == 200]
    if len(team100_puuids) != 5 or len(team200_puuids) != 5:
        raise HTTPException(400, "Could not split teams into 5v5.")

    try:
        feats: Dict[str, float] = runtime_team_deltas_from_puuids(
            team100_puuids, team200_puuids, window=window, history_count=history_count
        )
    except Exception as e:
        raise HTTPException(400, f"feature computation failed: {e}")

    try:
        teamA_ids = [puuid_to_riot_id(p) for p in team100_puuids]
        teamB_ids = [puuid_to_riot_id(p) for p in team200_puuids]
    except Exception:
        teamA_ids = team100_puuids
        teamB_ids = team200_puuids

    model = getattr(app.state, "model", None)
    model_features = getattr(app.state, "model_features", None)
    if model is None or model_features is None:
        raise HTTPException(500, "Model not loaded on startup.")

    X = [[feats[name] for name in model_features]]
    prob = float(model.predict_proba(X)[:, 1])

    return {
        "match_id": match_id,
        "teamA": teamA_ids,
        "teamB": teamB_ids,
        "p_win_team100": round(prob, 4),
        "values": {k: round(feats[k], 4) for k in model_features},
        "params": {"history_count": history_count, "window": window}
    }

@app.get("/match_info/{match_id}")
def match_info(match_id: str):
    """summary of a match + handy external links."""
    try:
        m = _get_match(match_id)
    except Exception as e:
        raise HTTPException(404, f"Could not load match {match_id}: {e}")

    info = m.get("info", {})
    parts = info.get("participants", [])
    if len(parts) != 10:
        raise HTTPException(400, "Match does not have 10 participants.")

    start_ms = info.get("gameStartTimestamp")
    start_iso = None
    if start_ms:
        start_iso = datetime.datetime.utcfromtimestamp(start_ms/1000).isoformat() + "Z"

    duration_s = info.get("gameDuration", 0)
    queue_id = info.get("queueId")
    patch = info.get("gameVersion")
    map_id = info.get("mapId")

    roster = []
    for p in parts:
        try:
            rid = f"{p.get('riotIdGameName')}#{p.get('riotIdTagline')}" if p.get("riotIdGameName") else puuid_to_riot_id(p["puuid"])
        except Exception:
            rid = p["puuid"]
        name_for_url = rid.replace("#", "-").replace(" ", "%20")
        region = "na"
        roster.append({
            "teamId": p.get("teamId"),
            "win": p.get("win"),
            "riot_id": rid,
            "puuid": p.get("puuid"),
            "champion": p.get("championName"),
            "kda": f"{p.get('kills',0)}/{p.get('deaths',0)}/{p.get('assists',0)}",
            "opgg": f"https://www.op.gg/summoners/{region}/{name_for_url}",
        })

    log_match = _log_link(match_id, region_code="na")
    return {
        "match_id": match_id,
        "start_utc": start_iso,
        "duration_seconds": duration_s,
        "queueId": queue_id,
        "patch": patch,
        "mapId": map_id,
        "participants": roster,
        "external": {
            "leagueofgraphs": log_match,
            "note": "If LoG hasn’t indexed yet, open a participant’s OP.GG link and match by time."
        }
    }

@app.get("/recent_matches/riotid/{game}/{tag}")
def recent_matches_riotid(game: str, tag: str, count: int = 10, queue: Optional[int] = None):
    """
    Show a player's recent matches (newest first) with timestamps + handy links.
    Example:
      /recent_matches/riotid/T1%20thebaldffs/NA1?count=5&queue=420
    """
    try:
        puuid = riot_id_to_puuid(f"{game}#{tag}")
    except Exception as e:
        raise HTTPException(404, f"Could not resolve Riot ID {game}#{tag}: {e}")

    try:
        ms = fetch_recent_matches_for_puuid(puuid, count=max(1, count), queue_id=queue)
    except Exception as e:
        raise HTTPException(400, f"fetch failed: {e}")

    out = []
    for m in ms:
        info = m.get("info", {})
        mid = m["metadata"]["matchId"]
        start_ms = info.get("gameStartTimestamp")
        start_iso = datetime.datetime.utcfromtimestamp(start_ms/1000).isoformat() + "Z" if start_ms else None
        qid = info.get("queueId")
        out.append({
            "match_id": mid,
            "start_utc": start_iso,
            "queueId": qid,
            "duration_seconds": info.get("gameDuration", 0),
            "leagueofgraphs": _log_link(mid, region_code="na"),
        })

    out.sort(key=lambda r: r["start_utc"] or "", reverse=True)
    return {
        "riot_id": f"{game}#{tag}",
        "count": len(out),
        "queue_filter": queue,
        "matches": out
    }
