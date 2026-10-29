import pandas as pd
from pathlib import Path

files = sorted(Path("data/raw").rglob("RAW_House1_Part*.csv"))
parts = []
for f in files:
    d = pd.read_csv(f, usecols=[0, 1, 2], header=0)
    d.columns = ["time", "unix", "agg"]
    d["src"] = f.name
    parts.append(d)
raw = pd.concat(parts, ignore_index=True)
raw["ts"] = pd.to_datetime(raw["unix"], unit="s", utc=True)

# 1. Is the Time column UTC or UK local time?
t = pd.to_datetime(raw["time"], errors="coerce")
off = ((t - raw["ts"].dt.tz_localize(None)).dt.total_seconds() / 3600).round()
print("Time minus Unix (hours):\n", off.value_counts(dropna=False), "\n")

# 2. Where do duplicate timestamps come from?
dup = raw[raw.duplicated("unix", keep=False)]
print("rows in duplicated timestamps:", len(dup))
print("top dates:\n", dup["ts"].dt.date.value_counts().head(10))
g = dup.groupby("unix")["src"].nunique()
print("timestamps spanning both files:", (g > 1).sum(), "| inside one file:", (g == 1).sum(), "\n")

# 3. Does cleaned House 1 line up with raw House 1, month by month?
cl_df = pd.read_parquet("data/processed/house_1_1min.parquet")
cl = cl_df["Aggregate"] if "Aggregate" in cl_df.columns else cl_df["aggregate"]
rr = raw.drop_duplicates("unix").set_index("ts")["agg"].sort_index().resample("1min").mean()
res = {}
for s in (-120, -60, 0, 60, 120):
    x = pd.concat([cl.rename("cl"), rr.shift(s, freq="min").rename("raw")], axis=1).dropna()
    res[s] = x.groupby(x.index.to_period("M")).apply(lambda g: g["cl"].corr(g["raw"]))
out = pd.DataFrame(res).round(3)
out["best_shift_min"] = out.idxmax(axis=1)
print(out)
