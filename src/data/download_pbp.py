from pathlib import Path
import nfl_data_py as nfl

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    seasons = list(range(2016, 2025))  # 2016-2024 inclusive
    print(f"Downloading pbp for seasons: {seasons[0]}-{seasons[-1]} ...")

    pbp = nfl.import_pbp_data(seasons)

    out_path = RAW_DIR / "pbp_2016_2024.parquet"
    pbp.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(pbp):,} cols={pbp.shape[1]}")

if __name__ == "__main__":
    main()
