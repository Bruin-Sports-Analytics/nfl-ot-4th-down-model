# src/data/export_parquets_to_csv.py
from __future__ import annotations

from pathlib import Path
import argparse
import polars as pl

DEFAULT_PARQUETS = [
    "fourth_down_decisions.parquet",
    "team_season_4th_rates.parquet",
    "drive_summary.parquet",
    "team_rolling_offense.parquet",
    "fourth_down_with_features.parquet",
    "ot_first_possession_4th.parquet",
]

def export_one(
    parquet_path: Path,
    out_dir: Path,
    sample_rows: int,
    full_max_rows: int,
    seed: int,
) -> None:
    if not parquet_path.exists():
        print(f"SKIP (missing): {parquet_path}")
        return

    lf = pl.scan_parquet(str(parquet_path))
    n_rows = lf.select(pl.len()).collect().item()

    base = parquet_path.stem
    out_sample = out_dir / f"{base}_sample.csv"
    out_full = out_dir / f"{base}.csv"

    # Always export a sample (head or random sample)
    if sample_rows <= 0:
        print(f"Sample rows <= 0, skipping sample for {parquet_path.name}")
    else:
        if n_rows <= sample_rows:
            # small enough: export all as sample
            df_s = lf.collect()
        else:
            # random sample is nicer than head for inspection
            # Polars sample works on eager DF, so collect only the needed columns/rows
            df_s = lf.collect().sample(n=sample_rows, seed=seed, shuffle=True)
        df_s.write_csv(out_sample)
        print(f"WROTE sample: {out_sample.name}  ({min(n_rows, sample_rows):,} rows)")

    # Export full only if under threshold
    if n_rows <= full_max_rows:
        # streaming-ish: read fully (still can be big), but threshold keeps it sane
        lf.collect().write_csv(out_full)
        print(f"WROTE full:   {out_full.name}  ({n_rows:,} rows)")
    else:
        print(f"SKIP full:   {parquet_path.name} too large ({n_rows:,} rows > {full_max_rows:,})")

def main():
    parser = argparse.ArgumentParser(description="Export processed parquets to viewable CSVs (sample + optional full).")
    parser.add_argument("--processed_dir", default="data/processed", help="Where the parquet files live")
    parser.add_argument("--out_dir", default="data/view", help="Where to write CSVs")
    parser.add_argument("--sample_rows", type=int, default=50_000, help="Rows to export for each *_sample.csv")
    parser.add_argument("--full_max_rows", type=int, default=250_000, help="Only export full CSV if <= this many rows")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--files", nargs="*", default=DEFAULT_PARQUETS, help="Specific parquet filenames to export")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processed dir: {processed_dir.resolve()}")
    print(f"Output dir:    {out_dir.resolve()}")
    print(f"sample_rows={args.sample_rows:,}  full_max_rows={args.full_max_rows:,}\n")

    for name in args.files:
        export_one(processed_dir / name, out_dir, args.sample_rows, args.full_max_rows, args.seed)

    print("\nDone.")

if __name__ == "__main__":
    main()
