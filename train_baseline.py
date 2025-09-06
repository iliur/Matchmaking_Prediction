"""
Train a baseline P(win) model on the dataset built from features.py:
- Uses logistic regression
- Saves model + feature list to 'model.pkl'
"""

from __future__ import annotations
import argparse
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from features import build_dataset_for_riot_ids, FEATURE_NAMES

USE_HGB = False  # set True to try tree model + calibration

def time_split(df: pd.DataFrame, frac: float = 0.8):
    df = df.sort_values("start_ms").reset_index(drop=True)
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--riot-ids", type=str, required=True,
                        help="Comma-separated Riot IDs, e.g. 'T1 thebaldffs#NA1,RLAero#NA1'")
    parser.add_argument("--per-player", type=int, default=120, help="Recent matches to pull per seed player")
    parser.add_argument("--out", type=str, default="model.pkl", help="Output model path")
    args = parser.parse_args()

    riot_ids = [x.strip() for x in args.riot_ids.split(",") if x.strip()]
    print(f"[train] seeds = {riot_ids}")
    print(f"[train] per-player = {args.per_player}")

    # 1) Build dataset
    df = build_dataset_for_riot_ids(riot_ids, per_player_matches=args.per_player)
    if len(df) < 100:
        print(f"[warn] small dataset: {len(df)} rows")

    # 2) Train/val split (time based)
    tr, te = time_split(df, frac=0.8)
    print(f"[train] split: train={len(tr)}, test={len(te)}")

    Xtr = tr[FEATURE_NAMES].values
    ytr = tr["win"].values
    Xte = te[FEATURE_NAMES].values
    yte = te["win"].values

    # 3) Fit model
    if USE_HGB:
        base = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.06)
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        print("[train] model: HGB + isotonic calibration")
    else:
        model = LogisticRegression(max_iter=500)
        print("[train] model: LogisticRegression")

    model.fit(Xtr, ytr)
    print("[train] model fitted.")

    # 4) Evaluate
    p = model.predict_proba(Xte)[:, 1]
    print("[metrics] rows :", len(df))
    print("[metrics] AUC  :", roc_auc_score(yte, p))
    print("[metrics] LogLoss:", log_loss(yte, p))
    print("[metrics] Brier:", brier_score_loss(yte, p))

    # 5) Save model bundle
    bundle = dict(model=model, feature_names=FEATURE_NAMES, meta=dict(rows=len(df)))
    joblib.dump(bundle, args.out)
    print(f"[save] wrote {args.out}")

if __name__ == "__main__":
    main()
