import pandas as pd
import numpy as np

# Load data
file_path = r"C:\Users\bramj\OneDrive\Pictures\Documents\tyroid.csv"
df = pd.read_csv(file_path)

# Separate numeric and categorical
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Impute numeric columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        if df[col].skew() < 1:  # No strong skew → mean
            df[col].fillna(df[col].mean(), inplace=True)
        else:  # Skewed → median
            df[col].fillna(df[col].median(), inplace=True)

# Impute categorical columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\n✅ Missing values imputed successfully.")
print(df.isnull().sum())