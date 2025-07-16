import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = r"C:\Users\bramj\OneDrive\Pictures\Documents\tyroid.csv"
df = pd.read_csv(file_path)

# ─── 1. View column names and basic info ───
print("\n[Basic Info and Data Types]")
print(df.info())
print("\n[First 5 Rows of Data]")
print(df.head())

# ─── 2. Identify Categorical & Numeric Data ───
print("\n[Categorical Columns]")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(categorical_cols)

print("\n[Numeric Columns]")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(numeric_cols)

# ─── 3. Check value distribution for Categorical Columns ───
print("\n[Value Counts for Categorical Variables]")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ─── 4. Suggest Encoding ───
print("\n[Encoding Suggestions]")
for col in categorical_cols:
    unique_vals = df[col].dropna().unique()
    print(f"{col}: {len(unique_vals)} unique values → {'Ordinal (Label Encoding)' if 'low' in unique_vals or 'high' in unique_vals else 'Nominal (One-Hot Encoding)'}")

# ─── 5. Range for Numeric Columns ───
print("\n[Range of Numeric Columns]")
for col in numeric_cols:
    print(f"{col}: Min = {df[col].min()}, Max = {df[col].max()}")

# ─── 6. Missing Values ───
print("\n[Missing Values in Each Column]")
print(df.isnull().sum())

# ─── 7. Outlier Detection (Boxplot) ───
print("\n[Outlier Detection Using Boxplots]")
for col in numeric_cols:
    plt.figure(figsize=(5, 1.5))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# ─── 8. Mean & Std Dev of Numeric Variables ───
print("\n[Mean and Std Dev of Numeric Columns]")
for col in numeric_cols:
    mean = df[col].mean()
    std_dev = df[col].std()
    print(f"{col}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")