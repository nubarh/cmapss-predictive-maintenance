import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

DATA_PATH = "data/processed/train_fd001_processed.csv"

df = pd.read_csv(DATA_PATH)

# Features/target
y = df["RUL"].values
X = df.drop(columns=["RUL"])

# Keep engine_id only for splitting, not as feature
groups = X["engine_id"].values
X = X.drop(columns=["engine_id"])

# Train/test split by engine (IMPORTANT)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
# Error analysis by life stage
stages = [
    ("late (RUL<=20)", y_test <= 20),
    ("mid (20<RUL<=60)", (y_test > 20) & (y_test <= 60)),
    ("early (RUL>60)", y_test > 60),
]

print("\nMAE by life stage:")
for name, mask in stages:
    if mask.sum() == 0:
        continue
    stage_mae = mean_absolute_error(y_test[mask], pred[mask])
    print(f"{name}: {stage_mae:.3f}  (n={mask.sum()})")
