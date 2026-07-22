"""
train.py
--------
Trains on older seasons, tests on the most recent ones (never a random
split for time-series). Reports accuracy + log-loss against three baselines:
  1. Always pick the higher-Elo player
  2. Always pick the higher-ranked player
  3. Logistic regression
  4. Gradient boosting (XGBoost)
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

FEATURES = ["elo_diff", "surf_elo_diff", "rank_diff", "rank_pts_diff",
            "age_diff", "ht_diff", "form_diff", "h2h_diff", "exp_diff"]
TEST_FROM = 20230101   # everything on/after this date is the test set

df = pd.read_csv("features.csv")
train = df[df.tourney_date < TEST_FROM]
test = df[df.tourney_date >= TEST_FROM]
Xtr, ytr = train[FEATURES], train.target
Xte, yte = test[FEATURES], test.target
print(f"train={len(train)}  test={len(test)}  test span={test.tourney_date.min()}-{test.tourney_date.max()}\n")

# --- baselines (elo_diff / rank_diff already signed toward player1) ---
b_elo = (test.elo_diff > 0).astype(int)
b_rank = (test.rank_diff > 0).astype(int)
print(f"Baseline  higher-Elo   acc={accuracy_score(yte, b_elo):.4f}")
print(f"Baseline  higher-rank  acc={accuracy_score(yte, b_rank):.4f}\n")

# --- logistic regression ---
logit = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000))
logit.fit(Xtr, ytr)
p_lr = logit.predict_proba(Xte)[:, 1]
print(f"LogReg    acc={accuracy_score(yte, (p_lr>.5).astype(int)):.4f}  "
      f"logloss={log_loss(yte, p_lr):.4f}  brier={brier_score_loss(yte, p_lr):.4f}")

# --- gradient boosting ---
xgb = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=5, eval_metric="logloss", n_jobs=4)
xgb.fit(Xtr.fillna(Xtr.median()), ytr)
p_xgb = xgb.predict_proba(Xte.fillna(Xtr.median()))[:, 1]
print(f"XGBoost   acc={accuracy_score(yte, (p_xgb>.5).astype(int)):.4f}  "
      f"logloss={log_loss(yte, p_xgb):.4f}  brier={brier_score_loss(yte, p_xgb):.4f}\n")

# --- feature importance (gain) ---
imp = sorted(zip(FEATURES, xgb.feature_importances_), key=lambda x: -x[1])
print("XGB importance:", "  ".join(f"{k}={v:.2f}" for k, v in imp))

# --- calibration check: do predicted probs match actual win rates? ---
print("\nCalibration (XGBoost):")
bins = pd.cut(p_xgb, [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
cal = pd.DataFrame({"p": p_xgb, "y": yte.values}).groupby(bins, observed=True)
for interval, g in cal:
    print(f"  pred {interval}: mean_pred={g.p.mean():.3f}  actual={g.y.mean():.3f}  n={len(g)}")

pickle.dump({"model": xgb, "features": FEATURES,
             "median": Xtr.median().to_dict()}, open("model.pkl", "wb"))
print("\nsaved model.pkl")
