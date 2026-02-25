"""
Fix modeling_dataset.csv:
  1. Add posteam to drive_summary merge key (prevents BUF/NYJ fixed_drive collision)
  2. Bin ydstogo and yardline_100 before merging team_season_4th_rates
  3. Merge offense rates for posteam + defense rates for defteam separately
  4. Drop redundant constant columns (down, score_differential, first_ot_drive)
"""
import pandas as pd
import numpy as np
import os

BASE = "C:/Users/buike/Downloads/bsa/nfl-ot-4th-down-model/data/view"


def make_ydstogo_bin(x):
    if pd.isna(x):
        return np.nan
    x = int(x)
    if x == 1:
        return "1"
    elif x == 2:
        return "2"
    elif 3 <= x <= 5:
        return "3-5"
    elif 6 <= x <= 10:
        return "6-10"
    else:
        return "11+"


def make_yardline_bin(x):
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 20:
        return "0-20"
    elif x <= 40:
        return "21-40"
    elif x <= 60:
        return "41-60"
    elif x <= 80:
        return "61-80"
    else:
        return "81-100"


# Load cleaned files
ot    = pd.read_csv(os.path.join(BASE, "cleaned_ot_first_possession_4th.csv"))
ds    = pd.read_csv(os.path.join(BASE, "cleaned_drive_summary.csv"))
roll  = pd.read_csv(os.path.join(BASE, "cleaned_team_rolling_offense.csv"))
rates = pd.read_csv(os.path.join(BASE, "cleaned_team_season_4th_rates.csv"))

print("Base OT shape:", ot.shape)

# Drop constant / redundant columns before building model dataset
# - 'down' is always 4 (constant)
# - 'score_differential' is always 0 in OT first possession (constant)
# - 'first_ot_drive' is identical to 'fixed_drive' (redundant)
drop_const = [c for c in ["down", "score_differential", "first_ot_drive"] if c in ot.columns]
ot = ot.drop(columns=drop_const)
print("Dropped constant/redundant columns:", drop_const)

# ---- Merge 1: drive_summary (add posteam to key to fix BUF/NYJ collision) ----
ds_cols = ["game_id", "fixed_drive", "posteam",
           "drive_epa", "drive_success_rate",
           "drive_posteam_score_start", "drive_posteam_score_end",
           "drive_start_yard_line", "fixed_drive_result", "drive_points"]
ot = ot.merge(ds[ds_cols], on=["game_id", "fixed_drive", "posteam"], how="left")
print("After drive_summary merge:", ot.shape)
assert len(ot) == 96, f"Expected 96 rows after drive_summary merge, got {len(ot)}"

# ---- Merge 2: team rolling offense ----
roll = roll.rename(columns={"team": "posteam"})
roll_cols = ["game_id", "posteam", "epa_per_drive_roll6", "success_rate_roll6", "points_per_drive_roll6"]
ot = ot.merge(roll[roll_cols], on=["game_id", "posteam"], how="left")
print("After team_rolling_offense merge:", ot.shape)
assert len(ot) == 96, f"Expected 96 rows after rolling merge, got {len(ot)}"

# ---- Create bin columns on OT data ----
ot["ydstogo_bin"]  = ot["ydstogo"].apply(make_ydstogo_bin)
ot["yardline_bin"] = ot["yardline_100"].apply(make_yardline_bin)

print("\nydstogo_bin distribution:")
print(ot["ydstogo_bin"].value_counts().to_string())
print("\nyardline_bin distribution:")
print(ot["yardline_bin"].value_counts().to_string())

# ---- Merge 3a: offense rates for posteam ----
rates_off = (rates[rates["side"] == "offense"]
             .drop(columns=["side", "go_faced", "go_stop_rate"])
             .rename(columns={"team": "posteam",
                              "go_attempts": "off_go_attempts",
                              "go_conv_rate": "off_go_conv_rate"}))
ot = ot.merge(rates_off, on=["season", "posteam", "ydstogo_bin", "yardline_bin"], how="left")
print("\nAfter offense rates merge:", ot.shape)

# ---- Merge 3b: defense rates for defteam ----
rates_def = (rates[rates["side"] == "defense"]
             .drop(columns=["side", "go_attempts", "go_conv_rate"])
             .rename(columns={"team": "defteam",
                              "go_faced": "def_go_faced",
                              "go_stop_rate": "def_go_stop_rate"}))
ot = ot.merge(rates_def, on=["season", "defteam", "ydstogo_bin", "yardline_bin"], how="left")
print("After defense rates merge:", ot.shape)
assert len(ot) == 96, f"Expected 96 rows after rates merges, got {len(ot)}"

# ---- Final leakage sweep ----
LEAKAGE = {"final_score", "win_loss", "winner", "loser", "yards_gained",
           "go_converted", "go_failed", "fg_made", "fourth_down_converted",
           "fourth_down_failed", "field_goal_result", "wpa", "touchback",
           "qb_dropback", "qb_scramble"}
leak_found = [c for c in ot.columns if c in LEAKAGE]
if leak_found:
    print("WARNING: removing residual leakage:", leak_found)
    ot = ot.drop(columns=leak_found)
else:
    print("\nNo leakage columns found.")

# ---- Dedup ----
before = len(ot)
ot = ot.drop_duplicates()
print("Duplicate rows removed:", before - len(ot))

# ---- Report ----
print("\nFinal modeling_dataset shape:", ot.shape)
print("Columns:", list(ot.columns))

print("\nTarget 'decision' distribution:")
print(ot["decision"].value_counts(dropna=False).to_string())

print("\nMissing values in modeling dataset:")
miss = ot.isna().sum()
miss = miss[miss > 0]
if miss.empty:
    print("  none")
else:
    for c in miss.index:
        print(f"  {c}: {miss[c]} ({miss[c]/len(ot)*100:.1f}%)")

# ---- Save ----
out_path = os.path.join(BASE, "modeling_dataset.csv")
ot.to_csv(out_path, index=False)
print("\nSaved modeling_dataset.csv")

# ---- Append fix summary to cleaning_report.txt ----
note = [
    "",
    "=" * 60,
    "7. MODELING DATASET FIX NOTES",
    "=" * 60,
    "  Issue 1: drive_summary merge produced 97 rows (96 expected).",
    "    Cause: game 2023_01_BUF_NYJ has fixed_drive=22 for BOTH BUF and NYJ.",
    "    Fix: added 'posteam' as merge key -> 96 rows confirmed.",
    "",
    "  Issue 2: team_season_4th_rates merge fanned out (96->1298 rows).",
    "    Cause: ydstogo_bin/yardline_bin absent from OT table;",
    "           merge used only season+posteam, causing cartesian product.",
    "    Fix: derived ydstogo_bin and yardline_bin from raw values;",
    "         merged offense rates on (season, posteam, ydstogo_bin, yardline_bin);",
    "         merged defense rates on (season, defteam, ydstogo_bin, yardline_bin).",
    "",
    "  Issue 3: go_faced / go_stop_rate were 100% null after offense-side merge.",
    "    Cause: these are defense-side metrics in team_season_4th_rates.",
    "    Fix: added separate defense-side merge for defteam.",
    "",
    "  Constant columns removed from modeling dataset (non-informative):",
    "    down (always 4), score_differential (always 0 in OT), first_ot_drive (== fixed_drive).",
    "",
    f"  Final modeling_dataset: {ot.shape[0]} rows x {ot.shape[1]} cols",
    "",
    "  Rolling metrics (epa_per_drive_roll6, success_rate_roll6, points_per_drive_roll6):",
    "    ~38% missing -- expected for early-season games (no prior 6-game history).",
    "    Recommend imputation strategy decided at model training time.",
    "",
    "  off_go_conv_rate, off_go_attempts:",
    "    ~34% missing -- sparse historical 4th-down data for some team/bin combos.",
    "    Recommend treating as 'no prior data' with a prior or encoding as missing category.",
]
with open(os.path.join(BASE, "cleaning_report.txt"), "a", encoding="utf-8") as f:
    f.write("\n".join(note))
print("Appended fix notes to cleaning_report.txt")
print("\n>>> Modeling dataset fix complete.")
