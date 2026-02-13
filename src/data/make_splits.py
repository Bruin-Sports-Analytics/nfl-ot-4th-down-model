from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/fourth_downs.parquet")
OUT_PATH = Path("data/processed/fourth_downs_with_split.parquet")

TRAIN_FRAC = 0.75
SEED = 42

def main():
    df = pd.read_parquet(IN_PATH)

    rng = np.random.default_rng(SEED)
    df["split"] = "test"

    # split within each season, by game_id (prevents leakage)
    for season, g in df.groupby("season"):
        game_ids = g["game_id"].dropna().unique()
        rng.shuffle(game_ids)

        n_train = max(1, int(len(game_ids) * TRAIN_FRAC))
        train_games = set(game_ids[:n_train])

        df.loc[(df["season"] == season) & (df["game_id"].isin(train_games)), "split"] = "train"

    df.to_parquet(OUT_PATH, index=False)
    print(df["split"].value_counts())
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
