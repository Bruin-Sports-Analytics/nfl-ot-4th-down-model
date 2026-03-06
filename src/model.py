"""
model.py
========
Trains XGBoost to predict 4th down conversion probability.

Key fix: monotone_constraints=(-1,...) on ydstogo forces the model to always
predict lower probability as yards-to-go increases, preventing the model from
learning a spurious positive relationship via go_conv_rate confounding.

Run order:
    1. python src/build_features.py
    2. python src/model.py
    3. streamlit run src/app.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
BASE_DIR  = Path(r"C:\Users\andre\OneDrive\Desktop\BSA 4th Down Model")
DATA_PATH = BASE_DIR / "data" / "fourth_down_enhanced.csv"
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. Load
# =============================================================================
print("=" * 60)
print("Loading enhanced dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape   : {df.shape}")
print(f"  Seasons : {sorted(df['season'].unique())}")

# =============================================================================
# 2. Filter: go-for-it attempts only, ydstogo <= 10
# =============================================================================
go_mask = (df["go_converted"].astype(bool) == True) | \
          (df["go_failed"].astype(bool)    == True)
go = df[go_mask].copy()
print(f"\nGO-for-it attempts (all)     : {len(go)}")
go = go[go["ydstogo"] <= 10].copy()
print(f"GO-for-it attempts (<=10 yds): {len(go)}")

go["target"] = go["go_converted"].fillna(0).astype(int)
print("\nTarget distribution:")
print(go["target"].value_counts().rename({0: "Failed", 1: "Converted"}))

# =============================================================================
# 3. Features
#
# Feature order matters for monotone_constraints below.
# ydstogo MUST be index 0.
# =============================================================================
FEATURES = [
    "ydstogo",              # index 0 — monotone constraint: must DECREASE prob
    "season",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "temp",
    "shotgun",
    "epa_per_game_roll15",
    "success_rate_roll15",
    "points_per_game_roll15",
    "go_conv_rate",
    "go_stop_rate",
    "def_epa_per_game_roll15",
    "def_success_rate_roll15",
    "def_points_per_game_roll15",
]

go_model = go[FEATURES + ["target"]].copy()
rows_before = len(go_model)
go_model = go_model.dropna()
go_model["target"] = go_model["target"].astype(int)

print(f"\nRows after dropna : {len(go_model)}  "
      f"(dropped {rows_before - len(go_model)} rows)")
print(f"Class balance     : {go_model['target'].value_counts().to_dict()}")

# =============================================================================
# 4. Train / test split
# =============================================================================
train = go_model[go_model["season"] <= 2021]
test  = go_model[go_model["season"] >= 2022]

X_train = train.drop(columns=["target", "season"])
y_train = train["target"]
X_test  = test.drop(columns=["target", "season"])
y_test  = test["target"]

print(f"\nTrain (seasons <= 2021) : {X_train.shape[0]} rows")
print(f"Test  (seasons >= 2022) : {X_test.shape[0]} rows")

# =============================================================================
# 5. Train XGBoost with monotone constraint on ydstogo
#
# monotone_constraints: tuple matching feature column order in X_train
#   -1 = probability must decrease as feature increases (ydstogo)
#    0 = no constraint
#    1 = probability must increase as feature increases
#
# This directly fixes the bug where the model ignored ydstogo and predicted
# higher probabilities at longer distances due to go_conv_rate confounding.
# =============================================================================
print("\n" + "=" * 60)
print("Training XGBoost model (with monotone constraint on ydstogo) ...")

feature_cols  = X_train.columns.tolist()
ydstogo_idx   = feature_cols.index("ydstogo")
constraints   = [0] * len(feature_cols)
constraints[ydstogo_idx] = -1   # force: more yards = lower probability

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    monotone_constraints=tuple(constraints),
    random_state=42,
)
model.fit(X_train, y_train)

cv_logloss = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_log_loss")
print(f"5-Fold CV Log Loss (train): {-cv_logloss.mean():.4f} +/- {cv_logloss.std():.4f}")

# =============================================================================
# 6. Calibrate
# =============================================================================
print("Calibrating probabilities ...")
cal_model = CalibratedClassifierCV(model, method="isotonic", cv=5)
cal_model.fit(X_train, y_train)

# =============================================================================
# 7. Evaluate
# =============================================================================
y_prob = cal_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
baseline = np.full(len(y_test), y_train.mean())

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print(f"  Log Loss  (model)    : {log_loss(y_test, y_prob):.4f}")
print(f"  Log Loss  (baseline) : {log_loss(y_test, baseline):.4f}")
print(f"  Accuracy             : {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC              : {roc_auc_score(y_test, y_prob):.4f}")
print(f"  Brier Score          : {brier_score_loss(y_test, y_prob):.4f}")
print("=" * 60)

# =============================================================================
# 8. Sanity check — predictions should decrease with ydstogo
# =============================================================================
print("\nSanity check (average team stats, ydstogo 1-10):")
avg = X_train.mean()
for ytg in range(1, 11):
    row = avg.copy()
    row["ydstogo"] = ytg
    prob = cal_model.predict_proba(pd.DataFrame([row]))[0][1]
    print(f"  4th and {ytg:2d}: {prob:.1%}")

# =============================================================================
# 9. Save model
# =============================================================================
model_path = BASE_DIR / "data" / "conversion_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": cal_model, "features": X_train.columns.tolist()}, f)
print(f"\nModel saved -> {model_path}")

# =============================================================================
# 10. Plots
# =============================================================================
BLUE_DARK  = "#1d4ed8"
BLUE_LIGHT = "#93c5fd"
GREEN      = "#16a34a"
RED        = "#dc2626"
GRAY       = "#6b7280"

# Plot 1: Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(prob_pred, prob_true, marker="o", lw=2, color=BLUE_DARK, label="Calibrated Model")
ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve - 4th Down Conversion Model")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "plot1_calibration_curve.png", dpi=150)
plt.close(fig)
print("Saved: plot1_calibration_curve.png")

# Plot 2: Feature Importance
importances = model.feature_importances_
feat_names  = X_train.columns.tolist()
order       = np.argsort(importances)
bar_colors  = [BLUE_DARK if i == order[-1] else BLUE_LIGHT for i in order]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([feat_names[i] for i in order], importances[order], color=bar_colors)
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("Feature Importance - 4th Down Conversion Model (XGBoost)")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "plot2_feature_importance.png", dpi=150)
plt.close(fig)
print("Saved: plot2_feature_importance.png")

# Plot 3: Conversion Rate vs Yards To Go
tmp = test[["ydstogo", "target"]].copy()
tmp["ydstogo_int"] = tmp["ydstogo"].round().astype(int)
rate = (tmp.groupby("ydstogo_int")
           .agg(conv_rate=("target", "mean"), n=("target", "count"))
           .reset_index())
rate = rate[rate["n"] >= 10]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(rate["ydstogo_int"], rate["conv_rate"],
       color=BLUE_LIGHT, edgecolor=BLUE_DARK, linewidth=0.8)
for _, row in rate.iterrows():
    ax.text(row["ydstogo_int"], row["conv_rate"] + 0.015,
            f'n={int(row["n"])}', ha="center", fontsize=7.5, color=GRAY)
ax.set_xlabel("Yards To Go")
ax.set_ylabel("Conversion Rate")
ax.set_title("4th Down Conversion Rate vs Yards To Go (Test Set, ydstogo<=10, n>=10)")
ax.set_ylim(0, 1.1); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "plot3_conversion_by_ydstogo.png", dpi=150)
plt.close(fig)
print("Saved: plot3_conversion_by_ydstogo.png")

# Plot 4: Sanity check — predicted prob vs ydstogo for good/bad/avg teams
avg_stats = X_train.mean()
teams_to_plot = {"Strong Offense (top 25%)": 0.75, "Average": 0.5, "Weak Offense (bottom 25%)": 0.25}
epa_quantiles = X_train["epa_per_game_roll15"].quantile([0.25, 0.5, 0.75])

fig, ax = plt.subplots(figsize=(8, 5))
for label, q in [("Strong Offense", 0.75), ("Average Offense", 0.5), ("Weak Offense", 0.25)]:
    epa_val = X_train["epa_per_game_roll15"].quantile(q)
    probs = []
    for ytg in range(1, 11):
        row = avg_stats.copy()
        row["ydstogo"] = ytg
        row["epa_per_game_roll15"] = epa_val
        probs.append(cal_model.predict_proba(pd.DataFrame([row]))[0][1])
    ax.plot(range(1, 11), probs, marker="o", label=label)
ax.set_xlabel("Yards To Go")
ax.set_ylabel("Predicted Conversion Probability")
ax.set_title("Predicted Conversion Probability vs Yards To Go by Team Strength")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 11))
plt.tight_layout()
fig.savefig(OUT_DIR / "plot4_prob_vs_ydstogo_by_team.png", dpi=150)
plt.close(fig)
print("Saved: plot4_prob_vs_ydstogo_by_team.png")

print(f"\nDone. All outputs saved to: {OUT_DIR}")
print("Next step: streamlit run src/app.py")
