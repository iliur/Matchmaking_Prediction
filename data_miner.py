import os
from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError
from fastapi import FastAPI, HTTPException
from typing import Optional

load_dotenv()

API_KEY  = os.getenv("RIOT_API_KEY")
PLATFORM = os.getenv("RIOT_PLATFORM", "na1")
REGION   = os.getenv("RIOT_REGION", "americas")

if not API_KEY:
    raise RuntimeError("Missing RIOT_API_KEY in .env or environment")

lol  = LolWatcher(API_KEY)
riot = RiotWatcher(API_KEY)

app = FastAPI()

@app.get("/health/env")
def health_env():
    return {"platform": PLATFORM, "region": REGION, "has_key": bool(API_KEY)}

@app.get("/debug/riotid/{game}/{tag}")
def debug_riotid(game: str, tag: str):
    """Return raw Account-V1 + Summoner-V4 payloads to see what's missing."""
    try:
        acc = riot.account.by_riot_id(REGION, game, tag)
    except ApiError as e:
        raise HTTPException(e.response.status_code, f"account.by_riot_id error: {e}")

    try:
        summ = lol.summoner.by_puuid(PLATFORM, acc["puuid"])
    except ApiError as e:
        raise HTTPException(e.response.status_code, f"summoner.by_puuid error: {e}")

    return {"account": acc, "summoner": summ}

@app.get("/ranked/riotid/{game}/{tag}")
def ranked_summary(game: str, tag: str):
    """
    Call with: /ranked/riotid/T1%20thebaldffs/NA1   (NO '#' in tag; Swagger will encode the space)
    """
    try:
        acc = riot.account.by_riot_id(REGION, game, tag)
    except ApiError as e:
        raise HTTPException(e.response.status_code, f"account.by_riot_id error: {e}")

    try:
        summ = lol.summoner.by_puuid(PLATFORM, acc["puuid"])
    except ApiError as e:
        raise HTTPException(e.response.status_code, f"summoner.by_puuid error: {e}")

    summ_id = summ.get("id")
    if not summ_id:
        # Helpful message instead of KeyError
        raise HTTPException(
            404,
            f"Summoner profile not found for PUUID on platform '{PLATFORM}'. "
            f"Raw summoner response: {summ}"
        )

    try:
        entries = lol.league.by_summoner(PLATFORM, summ_id)
    except ApiError as e:
        raise HTTPException(e.response.status_code, f"league.by_summoner error: {e}")

    out = []
    for e in entries:
        wins = e.get("wins", 0); losses = e.get("losses", 0)
        games = wins + losses
        wr = round(100 * wins / games, 2) if games else 0.0
        out.append({
            "queueType": e.get("queueType"),
            "tier": e.get("tier"),
            "rank": e.get("rank"),
            "leaguePoints": e.get("leaguePoints"),
            "wins": wins,
            "losses": losses,
            "winrate_pct": wr,
        })
    return {"riot_id": f"{game}#{tag}", "puuid": acc["puuid"], "summonerId": summ_id, "ranked": out}


@app.get("/winrate/riotid/{game}/{tag}")
def recent_winrate(game: str, tag: str, count: int = 20, queue_id: Optional[int] = None):
    """
    Example:
      /winrate/riotid/T1%20thebaldffs/NA1?count=30&queue_id=420
    queue_id: 420 SoloQ, 440 Flex, 400/430 Normals, 450 ARAM.
    """
    try:
        acc = riot.account.by_riot_id(REGION, game, tag)
        puuid = acc["puuid"]

        mids = lol.match.matchlist_by_puuid(REGION, puuid, count=max(count, 30))

        wins = 0
        losses = 0
        used = 0

        for mid in mids:
            if used >= count:
                break
            m = lol.match.by_id(REGION, mid)
            info = m.get("info", {})

            if queue_id is not None and info.get("queueId") != queue_id:
                continue

            for p in info.get("participants", []):
                if p.get("puuid") == puuid:
                    if p.get("win"):
                        wins += 1
                    else:
                        losses += 1
                    used += 1
                    break

        games = wins + losses
        return {
            "riot_id": f"{game}#{tag}",
            "window_games": games,
            "wins": wins,
            "losses": losses,
            "winrate_pct": round(100 * wins / games, 2) if games else 0.0,
            "queue_id": queue_id,
        }
    except ApiError as e:
        raise HTTPException(e.response.status_code, str(e))