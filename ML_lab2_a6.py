import pandas as pd
from numpy import dot
from numpy.linalg import norm

# Load CSV file (not Excel)
file_path = r"C:\Users\bramj\OneDrive\Pictures\Documents\tyroid.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns
df_numeric = df.select_dtypes(include='number')

# Get first two observation vectors
vec1 = df_numeric.iloc[0].values
vec2 = df_numeric.iloc[1].values

# Compute Cosine Similarity
cosine_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))

print(f"Cosine Similarity between first two documents: {cosine_sim:.4f}")