import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data
file_path = r"C:\Users\bramj\OneDrive\Pictures\Documents\tyroid.csv"
df = pd.read_csv(file_path)

# Select numeric attributes
numeric_cols = df.select_dtypes(include='number').columns
df_numeric = df[numeric_cols].copy()

# Decide scaling method: MinMax if range matters, Standard if Gaussian
scaler = MinMaxScaler()  # or use StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=numeric_cols)

print("\n Normalized Data Preview:")
print(df_scaled.head())