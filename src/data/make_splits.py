import polars as pl

df = pl.read_parquet("data/processed/fourth_down_with_features.parquet")

print(df.columns)
print(df.shape)
df.head(10)
