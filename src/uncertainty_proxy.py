import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/processed/train_fd001_processed.csv"
N_RUNS = 10 
TEST_SIZE = 0.2

df = pd.read_csv(DATA_PATH)

y = df["RUL"].values
X = df.drop(columns=["RUL"])
groups = X["engine_id"].values
X = X.drop(columns=["engine_id"])

maes = []
all_preds = [] #Store predictions from multiple random splits to estimate prediction variability 
all_true = []

for seed in range(N_RUNS):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    maes.append(mean_absolute_error(y_test, pred))
    all_preds.append(pred)
    all_true.append(y_test)

maes = np.array(maes)

print(f"MAE over {N_RUNS} random engine splits:")
print(f"mean={maes.mean():.3f}  std={maes.std():.3f}  min={maes.min():.3f}  max={maes.max():.3f}")

# Simple uncertainty summary: average prediction std within each run's test set
pred_stds = [np.std(p) for p in all_preds]
print(f"Prediction spread (std of predictions) across test sets: mean={np.mean(pred_stds):.3f}")


