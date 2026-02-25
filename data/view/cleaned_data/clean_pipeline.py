import pandas as pd
import numpy as np
import os
import re

BASE = "C:/Users/buike/Downloads/bsa/nfl-ot-4th-down-model/data/view"
LOG  = open(os.path.join(BASE, "pipeline_output.log"), "w", encoding="utf-8")

def log(msg=""):
    try:
        print(msg)
    except Exception:
        pass
    LOG.write(str(msg) + "\n")
    LOG.flush()

FILES = {
    "drive_summary":             "drive_summary.csv",
    "fourth_down_decisions":     "fourth_down_decisions.csv",
    "fourth_down_with_features": "fourth_down_with_features.csv",
    "ot_first_possession":       "ot_first_possession_4th.csv",
    "team_rolling_offense":      "team_rolling_offense.csv",
    "team_season_4th_rates":     "team_season_4th_rates.csv",
}

LEAKAGE_COLS = {
    "final_score", "win_loss", "winner", "loser",
    "yards_gained", "go_converted", "go_failed", "fg_made",
    "fourth_down_converted", "fourth_down_failed", "field_goal_result",
    "wpa", "touchback", "qb_dropback", "qb_scramble",
}


def normalise_cols(df):
    df.columns = [re.sub(r"[^\w]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    return df


def pct_to_float(s):
    if s.dtype == object:
        mask = s.astype(str).str.contains("%", na=False)
        if mask.any():
            s = s.copy()
            s[mask] = s[mask].astype(str).str.replace("%", "", regex=False).astype(float) / 100
    return s


def bool_to_int(s):
    if s.dtype == object:
        lower = s.astype(str).str.lower().str.strip()
        uniq = set(lower.dropna().unique())
        if uniq.issubset({"true", "false"}):
            return lower.map({"true": 1, "false": 0})
    elif s.dtype == bool:
        return s.astype(int)
    return s


def iqr_flags(df, label):
    lines = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        bad = s[(s < lo) | (s > hi)]
        if not bad.empty:
            lines.append(
                "  IQR [" + col + "]: " + str(len(bad)) +
                " vals outside [" + str(round(lo, 2)) + "," + str(round(hi, 2)) +
                "] sample=" + str(bad.head(3).tolist())
            )
        if re.search(r"ydstogo|yards_to_go", col):
            neg = s[s < 0]
            if not neg.empty:
                lines.append("  DOMAIN [" + col + "]: " + str(len(neg)) + " negative ydstogo")
        if "yardline" in col:
            b = s[(s < 0) | (s > 100)]
            if not b.empty:
                lines.append("  DOMAIN [" + col + "]: " + str(len(b)) + " yardline outside [0,100]")
    for line in lines:
        log(line)
    return lines


def fill_missing(df, label):
    fill_log, high_miss = [], []
    for col in df.columns:
        rate = df[col].isna().mean()
        if rate == 0:
            continue
        s = df[col]
        bool_like = (
            s.dtype == bool or
            (s.dtype == object and
             set(s.dropna().astype(str).str.lower().str.strip().unique()).issubset({"true", "false", "0", "1"}))
        )
        is_num = pd.api.types.is_numeric_dtype(s) and not bool_like
        is_cat = s.dtype == object and not bool_like

        if bool_like or (is_num and s.dropna().nunique() <= 2):
            m = s.mode(dropna=True)
            fill = int(m.iloc[0]) if not m.empty else 0
            df[col] = s.fillna(fill)
            fill_log.append("  " + col + ": " + str(round(rate * 100, 1)) + "% -> mode(" + str(fill) + ") [binary]")
        elif is_num:
            if rate < 0.10:
                fill = s.median()
                df[col] = s.fillna(fill)
                fill_log.append("  " + col + ": " + str(round(rate * 100, 1)) + "% -> median(" + str(round(fill, 4)) + ")")
            elif rate <= 0.30:
                fill = s.mean()
                df[col] = s.fillna(fill)
                fill_log.append("  " + col + ": " + str(round(rate * 100, 1)) + "% -> mean(" + str(round(fill, 4)) + ")")
            else:
                high_miss.append(col)
                fill_log.append("  " + col + ": " + str(round(rate * 100, 1)) + "% -> FLAGGED >30% (not filled)")
        elif is_cat:
            df[col] = s.fillna("unknown")
            fill_log.append("  " + col + ": " + str(round(rate * 100, 1)) + "% -> unknown [categorical]")
    return df, fill_log, high_miss


def inspect(df, label):
    log()
    log("=" * 60)
    log(" " + label + "  |  shape=" + str(df.shape))
    log("=" * 60)
    log("Columns: " + str(list(df.columns)))
    log("Dtypes:")
    for c, dt in df.dtypes.items():
        log("  " + str(c) + ": " + str(dt))
    miss = df.isna().sum()
    miss = miss[miss > 0]
    log("Missing values (non-zero):")
    if miss.empty:
        log("  none")
    else:
        pct = (miss / len(df) * 100).round(1)
        for c in miss.index:
            log("  " + str(c) + ": " + str(miss[c]) + " (" + str(pct[c]) + "%)")
    log("Numeric summary (mean/median/min/max):")
    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna()
        if s.empty:
            continue
        log("  " + c + ": " + str(round(s.mean(), 3)) + " / " + str(round(s.median(), 3)) +
            " / " + str(round(s.min(), 3)) + " / " + str(round(s.max(), 3)))
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const:
        log("Constant cols: " + str(const))


# =============================================================================
dataframes = {}
leakage_dfs = {}
all_outliers = {}
all_fill = {}
high_miss_all = {}
type_log_all = {}

log("#" * 60)
log("  NFL OT 4th Down -- Cleaning Pipeline")
log("#" * 60)

# STEPS 1+2: Inspect and Clean each file
for key, fname in FILES.items():
    fpath = os.path.join(BASE, fname)
    log()
    log(">>> Loading " + fname)
    df = pd.read_csv(fpath, low_memory=False)

    # STEP 1 - Inspect
    inspect(df, key)

    # 2a - Column standardisation
    df = normalise_cols(df)

    # 2b - Type corrections
    tcor = []
    for col in df.select_dtypes("object").columns:
        new = pct_to_float(df[col])
        if not new.equals(df[col]):
            df[col] = new
            tcor.append("  " + col + ": pct->float")
    for col in df.columns:
        new = bool_to_int(df[col])
        if not new.equals(df[col]):
            df[col] = new
            tcor.append("  " + col + ": bool->int")
    force_num = [
        "ydstogo", "yards_to_go", "yardline_100", "score_differential",
        "game_seconds_remaining", "drive_epa", "epa", "wp",
        "epa_per_drive_roll6", "success_rate_roll6", "points_per_drive_roll6",
        "go_conv_rate", "go_stop_rate", "kick_distance",
        "drive_points", "drive_posteam_score_start", "drive_posteam_score_end",
    ]
    for col in force_num:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            tcor.append("  " + col + ": ->numeric")
    for col in ["qtr", "quarter", "season", "week", "down", "fixed_drive", "drive"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["posteam", "defteam", "home_team", "away_team", "team", "roof", "decision"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    type_log_all[key] = tcor
    if tcor:
        log()
        log("[" + key + "] Type corrections:")
        for t in tcor:
            log(t)

    # STEP 4 - Outlier flags
    log()
    log("[" + key + "] Outlier checks:")
    flags = iqr_flags(df, key)
    all_outliers[key] = flags
    if not flags:
        log("  none")

    # STEP 5 - Leakage detection
    found_leak = [c for c in df.columns if c in LEAKAGE_COLS]
    if found_leak:
        leakage_dfs[key] = df[found_leak].copy()
        df = df.drop(columns=found_leak)
        log()
        log("[" + key + "] Leakage cols removed: " + str(found_leak))

    # 2c - Missing value imputation
    df, fl, hm = fill_missing(df, key)
    all_fill[key] = fl
    if hm:
        high_miss_all[key] = hm
    if fl:
        log()
        log("[" + key + "] Missing value imputation:")
        for l in fl:
            log(l)
    else:
        log()
        log("[" + key + "] No missing values to fill.")

    dataframes[key] = df
    log()
    log("[" + key + "] Cleaned shape: " + str(df.shape))

# =============================================================================
# STEP 3 - Merge Consistency
# =============================================================================
log()
log("=" * 60)
log("  STEP 3 -- MERGE CONSISTENCY CHECK")
log("=" * 60)
KEY_COLS = ["game_id", "season", "week"]
core = ["drive_summary", "fourth_down_decisions", "fourth_down_with_features", "ot_first_possession"]
log("Merge keys present per file:")
for k in FILES:
    present = [c for c in KEY_COLS if c in dataframes[k].columns]
    log("  " + k + ": " + str(present))

gid = {}
for k in core:
    if "game_id" in dataframes[k].columns:
        gid[k] = set(dataframes[k]["game_id"].dropna().unique())
log("game_id unique counts:")
for k, s in gid.items():
    log("  " + k + ": " + str(len(s)))
ot_ids = gid.get("ot_first_possession", set())
for k, s in gid.items():
    if k == "ot_first_possession":
        continue
    miss_g = ot_ids - s
    log("  OT games not in " + k + ": " + str(len(miss_g)))
    if miss_g:
        log("    sample: " + str(list(miss_g)[:5]))

log("Duplicate rows:")
for k, df in dataframes.items():
    log("  " + k + ": " + str(df.duplicated().sum()) + " dupes")

ot_df = dataframes["ot_first_possession"]
ds_df = dataframes["drive_summary"]
if (all(c in ot_df.columns for c in ["game_id", "fixed_drive"]) and
        all(c in ds_df.columns for c in ["game_id", "fixed_drive"])):
    ot_pairs = set(zip(ot_df["game_id"], ot_df["fixed_drive"]))
    ds_pairs = set(zip(ds_df["game_id"], ds_df["fixed_drive"]))
    un = ot_pairs - ds_pairs
    log("OT (game_id,fixed_drive) not in drive_summary: " + str(len(un)))
    if un:
        log("  sample: " + str(list(un)[:5]))

# =============================================================================
# STEP 4 - Save cleaned CSVs
# =============================================================================
log()
log("=" * 60)
log("  STEP 4 -- SAVING CLEANED FILES")
log("=" * 60)
saved_shapes = {}
for key, df in dataframes.items():
    out = os.path.join(BASE, "cleaned_" + FILES[key])
    df.to_csv(out, index=False)
    saved_shapes[key] = df.shape
    log("  Saved cleaned_" + FILES[key] + "  " + str(df.shape))
for key, ldf in leakage_dfs.items():
    out = os.path.join(BASE, "leakage_" + FILES[key])
    ldf.to_csv(out, index=False)
    log("  Saved leakage_" + FILES[key] + "  " + str(ldf.shape))

# =============================================================================
# STEP 5 - Modeling dataset
# =============================================================================
log()
log("=" * 60)
log("  STEP 5 -- BUILDING modeling_dataset.csv")
log("=" * 60)

ot = dataframes["ot_first_possession"].copy()
log("Base (ot_first_possession): " + str(ot.shape))

ds = dataframes["drive_summary"].copy()
ds_extra = ["game_id", "fixed_drive"] + [c for c in ds.columns if c not in ot.columns]
ot = ot.merge(ds[ds_extra], on=["game_id", "fixed_drive"], how="left", suffixes=("", "_ds"))
log("After drive_summary merge: " + str(ot.shape))

roll = dataframes["team_rolling_offense"].copy()
if "team" in roll.columns:
    roll = roll.rename(columns={"team": "posteam"})
if "posteam" in roll.columns and "game_id" in roll.columns:
    roll_extra = ["game_id", "posteam"] + [c for c in roll.columns if c not in ot.columns]
    ot = ot.merge(roll[roll_extra], on=["game_id", "posteam"], how="left", suffixes=("", "_roll"))
    log("After team_rolling_offense merge: " + str(ot.shape))

rates = dataframes["team_season_4th_rates"].copy()
if "side" in rates.columns:
    rates = rates[rates["side"] == "offense"].drop(columns=["side"])
if "team" in rates.columns and "posteam" not in rates.columns:
    rates = rates.rename(columns={"team": "posteam"})
rkeys = [k for k in ["season", "posteam", "ydstogo_bin", "yardline_bin"]
         if k in rates.columns and k in ot.columns]
if rkeys:
    rcols = rkeys + [c for c in rates.columns if c not in rkeys]
    ot = ot.merge(rates[rcols], on=rkeys, how="left", suffixes=("", "_rates"))
    log("After team_season_4th_rates merge: " + str(ot.shape))

final_leak = [c for c in ot.columns if c in LEAKAGE_COLS]
if final_leak:
    log("WARN: removing residual leakage: " + str(final_leak))
    ot = ot.drop(columns=final_leak)
else:
    log("No leakage columns in modeling dataset.")

if "decision" in ot.columns:
    log("Target value counts:")
    log(ot["decision"].value_counts(dropna=False).to_string())
else:
    log("WARN: target column 'decision' not found!")

bef = len(ot)
ot = ot.drop_duplicates()
log("Duplicate rows removed: " + str(bef - len(ot)))
log("Final modeling_dataset shape: " + str(ot.shape))
log("Columns (" + str(len(ot.columns)) + "): " + str(list(ot.columns)))
ot.to_csv(os.path.join(BASE, "modeling_dataset.csv"), index=False)
log("Saved modeling_dataset.csv")

# =============================================================================
# Write cleaning_report.txt
# =============================================================================
rpt = []
rpt.append("NFL OT 4th Down Model -- Data Cleaning Report")
rpt.append("=" * 60)
rpt.append("Generated: 2026-02-19")
rpt.append("")
rpt.append("=" * 60)
rpt.append("1. MISSING VALUES HANDLED")
rpt.append("=" * 60)
for k, lines in all_fill.items():
    rpt.append("")
    rpt.append("[" + k + "]")
    rpt.extend(lines if lines else ["  (none)"])

rpt.append("")
rpt.append("=" * 60)
rpt.append("2. COLUMNS FLAGGED >30% MISSING (not auto-filled)")
rpt.append("=" * 60)
if high_miss_all:
    for k, cols in high_miss_all.items():
        rpt.append("  " + k + ": " + str(cols))
else:
    rpt.append("  None")

rpt.append("")
rpt.append("=" * 60)
rpt.append("3. LEAKAGE COLUMNS DETECTED & REMOVED")
rpt.append("=" * 60)
if leakage_dfs:
    for k, ldf in leakage_dfs.items():
        rpt.append("  " + k + ": " + str(list(ldf.columns)))
else:
    rpt.append("  None detected")

rpt.append("")
rpt.append("=" * 60)
rpt.append("4. TYPE CORRECTIONS")
rpt.append("=" * 60)
for k, lines in type_log_all.items():
    rpt.append("")
    rpt.append("[" + k + "]")
    rpt.extend(lines if lines else ["  (none)"])

rpt.append("")
rpt.append("=" * 60)
rpt.append("5. OUTLIERS FLAGGED (not removed)")
rpt.append("=" * 60)
any_out = False
for k, lines in all_outliers.items():
    if lines:
        rpt.append("")
        rpt.append("[" + k + "]")
        rpt.extend(lines)
        any_out = True
if not any_out:
    rpt.append("  None")

rpt.append("")
rpt.append("=" * 60)
rpt.append("6. FINAL CLEANED DATASET SHAPES")
rpt.append("=" * 60)
for k, sh in saved_shapes.items():
    rpt.append("  " + k + ": " + str(sh[0]) + " rows x " + str(sh[1]) + " cols")
rpt.append("  modeling_dataset: " + str(ot.shape[0]) + " rows x " + str(ot.shape[1]) + " cols")

with open(os.path.join(BASE, "cleaning_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(rpt))
log("Saved cleaning_report.txt")
log(">>> Pipeline complete.")
LOG.close()
