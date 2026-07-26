import pandas as pd
from src.features import build_features

# Load one customer from your training data
df = pd.read_csv("data/training/training.csv").head(3)
print("=== RAW (what came in) ===")
print(df[["customerID", "gender", "Contract", "tenure", "MonthlyCharges",
          "InternetService", "OnlineBackup", "StreamingTV"]])

# Transform
X = build_features(df)
print("\n=== FEATURES (what the model sees) ===")
print(X)

print(f"\nColumns in: {len(df.columns)}   Columns out: {len(X.columns)}")