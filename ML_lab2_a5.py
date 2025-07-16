import pandas as pd

# Load Excel file
file_path = r"C:\Users\bramj\OneDrive\Pictures\Documents\tyroid.csv"
df = pd.read_csv(file_path)

# Select first 2 rows
v1 = df.iloc[0]
v2 = df.iloc[1]

# Filter only binary columns
binary_cols = df.columns[df.apply(lambda col: set(col.dropna().unique()).issubset({0, 1}))]

# Get binary values from first two rows
vec1 = v1[binary_cols].astype(int).values
vec2 = v2[binary_cols].astype(int).values

# Count f11, f10, f01, f00
f11 = sum((vec1 == 1) & (vec2 == 1))
f10 = sum((vec1 == 1) & (vec2 == 0))
f01 = sum((vec1 == 0) & (vec2 == 1))
f00 = sum((vec1 == 0) & (vec2 == 0))

# Compute JC and SMC
jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else 0
smc = (f11 + f00) / (f11 + f10 + f01 + f00)

print(f"f11: {f11}, f10: {f10}, f01: {f01}, f00: {f00}")
print(f"Jaccard Coefficient: {jc:.4f}")
print(f"Simple Matching Coefficient: {smc:.4f}")
