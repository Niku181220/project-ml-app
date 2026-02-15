import os
import pandas as pd

df = pd.read_csv("data/wdbc.data", header=None)

# columns: id, diagnosis, 30 features
cols = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
df.columns = cols

# save final dataset as CSV in same data folder
os.makedirs("data", exist_ok=True)
df.to_csv("data/dataset.csv", index=False)

print("Shape:", df.shape)
print(df.head())

