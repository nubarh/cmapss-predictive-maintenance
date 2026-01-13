import pandas as pd

TRAIN_PATH = "data/raw/train_FD001.txt"

columns = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

train_df = pd.read_csv(
    TRAIN_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

max_cycle = train_df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]

train_df = train_df.merge(max_cycle, on="engine_id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
train_df.drop(columns=["max_cycle"], inplace=True)

print("Shape:", train_df.shape)
print(train_df.head())
print(train_df["RUL"].describe())
# Save processed data
OUTPUT_PATH = "data/processed/train_fd001.csv"

import os
os.makedirs("data/processed", exist_ok=True)

train_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved processed data to {OUTPUT_PATH}")

