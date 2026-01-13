import os
import pandas as pd

INPUT_PATH = "data/raw/train_FD001.txt"
OUTPUT_PATH = "data/processed/train_fd001_processed.csv"

columns = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(INPUT_PATH, sep=r"\s+", header=None, names=columns)

# Compute RUL
max_cycle = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df["engine_id"].map(max_cycle) - df["cycle"]
RUL_CAP = 125
df["RUL"] = df["RUL"].clip(upper=RUL_CAP)


# Drop constant sensors
sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
constant_sensors = df[sensor_cols].std()[lambda s: s == 0].index.tolist()
df = df.drop(columns=constant_sensors)

# Save
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Dropped constant sensors:", constant_sensors)
print("Saved:", OUTPUT_PATH)
print("Shape:", df.shape)
