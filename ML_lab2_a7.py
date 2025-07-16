import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm

# Load data
file_path = r"C:\Users\bramj\OneDrive\Pictures\Documents\tyroid.csv"
df = pd.read_csv(file_path)

# ─── Prepare Data ───
df_binary = df.select_dtypes(include='number').copy()
binary_cols = [col for col in df_binary.columns if set(df_binary[col].dropna().unique()).issubset({0, 1})]
df_binary = df_binary[binary_cols].iloc[:20]  # Use first 20 observations only

# ─── Initialize Similarity Matrices ───
n = len(df_binary)
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))
cos_matrix = np.zeros((n, n))

# ─── Compute JC, SMC, COS for Each Pair ───
for i in range(n):
    for j in range(n):
        vec1 = df_binary.iloc[i]
        vec2 = df_binary.iloc[j]

        f11 = np.sum((vec1 == 1) & (vec2 == 1))
        f00 = np.sum((vec1 == 0) & (vec2 == 0))
        f10 = np.sum((vec1 == 1) & (vec2 == 0))
        f01 = np.sum((vec1 == 0) & (vec2 == 1))

        jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
        smc = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0
        cos = dot(vec1, vec2) / (norm(vec1) * norm(vec2)) if norm(vec1) * norm(vec2) != 0 else 0

        jc_matrix[i][j] = jc
        smc_matrix[i][j] = smc
        cos_matrix[i][j] = cos

# ─── Plot Heatmaps ───
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
sns.heatmap(jc_matrix, annot=False, cmap='coolwarm')
plt.title("Jaccard Coefficient (JC)")

plt.subplot(1, 3, 2)
sns.heatmap(smc_matrix, annot=False, cmap='viridis')
plt.title("Simple Matching Coefficient (SMC)")

plt.subplot(1, 3, 3)
sns.heatmap(cos_matrix, annot=False, cmap='plasma')
plt.title("Cosine Similarity")

plt.tight_layout()
plt.show()