from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = Path("data/raw/pbp_2016_2024.parquet")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(RAW_PATH)

    # Regular season only
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    # All 4th downs (regulation + OT) -> for training the sub-models
    fourth = df[df["down"] == 4].copy()

    # Decision label
    punt = fourth.get("punt_attempt", 0).fillna(0).astype(int)
    fg = fourth.get("field_goal_attempt", 0).fillna(0).astype(int)

    fourth["decision"] = np.select(
        [punt == 1, fg == 1],
        ["punt", "fg"],
        default="go"
    )

    # "Go success" label (only meaningful if decision == go)
    fd = fourth.get("first_down", 0).fillna(0).astype(int)
    td = fourth.get("touchdown", 0).fillna(0).astype(int)
    fourth["go_success"] = ((fd == 1) | (td == 1)).astype(int)

    # Keep minimal set of columns (add more later)
    keep = [c for c in [
        "season","game_id","qtr","posteam","defteam",
        "yardline_100","ydstogo","score_differential",
        "game_seconds_remaining","half_seconds_remaining",
        "shotgun","wp","epa",
        "decision","go_success"
    ] if c in fourth.columns]

    fourth = fourth[keep].dropna(subset=["season","game_id","yardline_100","ydstogo"]).copy()

    out_path = OUT_DIR / "fourth_downs.parquet"
    fourth.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(fourth):,}")

if __name__ == "__main__":
    main()
