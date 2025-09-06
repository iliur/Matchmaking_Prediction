# Riot Games Matchmaking Prediction — P(win) Demo

Small system that predicts a League match’s **probability to win** using only public Riot APIs.  
It fetches players’ recent games, builds **recent-form, role-aware features**, learns from past matches, and serves **P(Team 100 wins)** via FastAPI.

---

## What it does (brief)
- **Pulls data** with `riotwatcher`: Riot-ID → **PUUID**, then **Match-V5** JSON for recent games.  
- **Caches** raw match JSON locally (`.riot_cache/`) to reduce API calls.  
- **Features (pre-match only; no leakage):** per player WR, KDA, CS/min, DPM, Kill-Participation, Role-consistency, and a numeric Rank; team **averages** + **TeamA − TeamB deltas**.  
- **Model:** **Logistic Regression** on the **delta features** to output calibrated-ish **P(win)**.

---

## Setup (quick)
1) **Python env**
```bash
conda create -n lol-pwin python=3.10 -y && conda activate lol-pwin
pip install -r requirements.txt
```

2. Secrets (.env) — do not commit
RIOT_API_KEY=replace_me
RIOT_PLATFORM=na1
RIOT_REGION=americas
MODEL_PATH=model.pkl
RIOT_CACHE_DIR=.riot_cache
