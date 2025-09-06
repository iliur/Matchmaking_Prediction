# Riot Matchmaking Prediction — P(win) Demo

---

## Brief
Small MVP to explore matchmaking algorithms and how machine learning can be applied in this field. This project can be expanded to anything to do with matchmaking even beyond League (I.e. Other Riot Games, Chess, Soccer (or any sports betting), etc)
League was only chosen due to Riot's easy-to-use API and largely available data.

---

## Intro (What it does)
Small system that predicts a League match’s **probability to win** using only public Riot APIs.  
It fetches players’ recent games, builds **recent-form, role-aware features**, learns from past matches, and serves **P(Team 100 wins)** via FastAPI.
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

---

## Run order (training → serving → demo)
1) Train a model

Build a dataset from one or more seed players (Riot IDs are GameName#Tag).
This pulls their recent SoloQ games, constructs features with time cutoffs, time-splits ~80/20, trains, prints AUC / LogLoss / Brier, and writes model.pkl.

python train_baseline.py \
  --riot-ids "cant type#1998,RLAero#NA1" \
  --per-player 80 \
  --out model.pkl

2) Start the API
export MODEL_PATH=model.pkl
python -m uvicorn app_scoring:app --reload

3) Demo endpoints

Health: GET /health

Recent matches for a player:
GET /recent_matches/riotid/cant%20type/1998?count=5&queue=420

Human summary (roster/champs/time/links):
GET /match_info/NA1_<MATCHID>

Predict from a real match id:
GET /pwin_from_match/NA1_<MATCHID>?history_count=60&window=20
→ returns p_win_team100 and the exact feature values used

What-if (hypothetical lobby):
POST /pwin with JSON:

{
  "teamA": ["name1#TAG","name2#TAG","name3#TAG","name4#TAG","name5#TAG"],
  "teamB": ["name6#TAG","name7#TAG","name8#TAG","name9#TAG","name10#TAG"]
}

---

## How data is pulled (in 4 steps)

Riot-ID → PUUID (Account-V1).

Recent match ids via Match-V5 (queue filter e.g. 420).

Per match, use gameStartTimestamp to select only earlier games for each player (pre-match stats: default window=20 from up to history_count=60).

Aggregate to team means and compute TeamA − TeamB deltas for the model.

---

## ML model (what & why)

Model: LogisticRegression(max_iter=500) on team-delta features.

Why LR?

Small/medium data → LR is stable, fast, and hard to overfit.

Explainable → each delta’s sign/magnitude shows why P(win) moved.

Probability quality → reasonably calibrated; simple to add Platt/Isotonic if needed.

Speed → sub-ms inference; great for live scoring.

Evaluation: time-based split (train first ~80% of matches, test on the next ~20%); report AUC / LogLoss / Brier.

Upgrade path: optional HistGradientBoosting + Isotonic (already in code) when you have more data or add richer features (champ/role embeddings, draft/synergy, queue/patch-specific models).

---

## TL;DR

Pull recent games → build recent-form features → train Logistic Regression on team deltas → return P(win) for a chosen match.

