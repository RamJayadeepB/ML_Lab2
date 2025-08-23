"""
23CSE301 – Lab 02 (Revised)
A1–A9 Modular Python Solutions (FINAL)
"""

import pandas as pd
import numpy as np
import statistics as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --------------------------
# Helper: Load Data
# --------------------------
def load_excel_data(excel, sheet):
    return pd.read_excel(excel, sheet_name=sheet)

# --------------------------
# A1: Linear Algebra
# --------------------------
def analyze_purchase_data(data):
    A = data.iloc[:, 1:4]  # Features (columns 1–3)
    C = data.iloc[:, 4:5]  # Target (column 4)
    dim = A.shape[1]
    num_vectors = A.shape[0]
    rank = np.linalg.matrix_rank(A)
    unit_costs = np.dot(np.linalg.pinv(A), C)
    return A, C, dim, num_vectors, rank, unit_costs

# --------------------------
# A2: Classification (RICH/POOR)
# --------------------------
def label_rich_poor(data, threshold=200):
    return np.where(data['Payment (Rs)'] > threshold, "RICH", "POOR")

def train_rich_poor_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# --------------------------
# A3: IRCTC Stock Analysis
# --------------------------
def stock_statistics(data):
    mean_price = st.mean(data['Price'])
    var_price = st.variance(data['Price'])
    return mean_price, var_price

def mean_by_day(data, day):
    day_data = data[data['Day'] == day]['Price']
    return st.mean(day_data) if not day_data.empty else np.nan

def mean_by_month(data, month):
    month_data = data[data['Month'] == month]['Price']
    return st.mean(month_data) if not month_data.empty else np.nan

def probability_of_loss(data):
    return (data['Chg%'] < 0).mean()

def probability_profit_wednesday(data):
    wed_data = data[data['Day'] == 'Wed']
    return (wed_data['Chg%'] > 0).mean() if not wed_data.empty else np.nan

def conditional_profit_given_wednesday(data):
    wed_data = data[data['Day'] == 'Wed']
    return (wed_data['Chg%'] > 0).sum() / len(wed_data) if len(wed_data) > 0 else np.nan

def plot_chg_vs_day(data):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data['Day'], y=data['Chg%'])
    plt.title("Change % vs Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.show()

# --------------------------
# A4: Thyroid Data Exploration
# --------------------------
def thyroid_exploration(th_data):
    numerical_cols = th_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = th_data.select_dtypes(include=['object', 'bool']).columns.tolist()
    missing_values = th_data.isnull().sum()
    return numerical_cols, categorical_cols, missing_values

# --------------------------
# A5: JC and SMC
# --------------------------
def jaccard(B1, B2):
    f11 = sum(a == 1 and b == 1 for a, b in zip(B1, B2))
    f10 = sum(a == 1 and b == 0 for a, b in zip(B1, B2))
    f01 = sum(a == 0 and b == 1 for a, b in zip(B1, B2))
    denom = f11 + f10 + f01
    return f11 / denom if denom > 0 else np.nan

def smc(B1, B2):
    f11 = sum(a == 1 and b == 1 for a, b in zip(B1, B2))
    f00 = sum(a == 0 and b == 0 for a, b in zip(B1, B2))
    f10 = sum(a == 1 and b == 0 for a, b in zip(B1, B2))
    f01 = sum(a == 0 and b == 1 for a, b in zip(B1, B2))
    return (f11 + f00) / (f11 + f00 + f10 + f01)

# --------------------------
# A6: Cosine Similarity
# --------------------------
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = np.sqrt(sum(a ** 2 for a in v1))
    norm2 = np.sqrt(sum(b ** 2 for b in v2))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

# --------------------------
# A7: Heatmaps
# --------------------------
def compute_heatmaps(df, n=20):
    df = df.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)
    binary_df = (
        df.replace({'t': 1, 'f': 0, '?': np.nan})
          .apply(pd.to_numeric, errors='coerce')
          .fillna(0)
          .astype(int)
          .iloc[:n]
    )

    JC = pd.DataFrame([[jaccard(binary_df.iloc[i], binary_df.iloc[j]) for j in range(n)] for i in range(n)])
    SMC = pd.DataFrame([[smc(binary_df.iloc[i], binary_df.iloc[j]) for j in range(n)] for i in range(n)])
    COS = pd.DataFrame([[cosine_similarity(binary_df.iloc[i], binary_df.iloc[j]) for j in range(n)] for i in range(n)])

    plt.figure(figsize=(30, 8))
    plt.subplot(1, 3, 1)
    sns.heatmap(JC, cmap='Blues')
    plt.title('Jaccard Coefficient')

    plt.subplot(1, 3, 2)
    sns.heatmap(SMC, cmap='Greens')
    plt.title('SMC')

    plt.subplot(1, 3, 3)
    sns.heatmap(COS, cmap='Oranges')
    plt.title('Cosine Similarity')

    plt.tight_layout()
    plt.show()

    return JC, SMC, COS

# --------------------------
# A8: Imputation
# --------------------------
def impute_data(th_data, use_mean, use_median, use_mode):
    for col in use_mean:
        th_data[col] = th_data[col].fillna(th_data[col].mean())
    for col in use_median:
        th_data[col] = th_data[col].fillna(th_data[col].median())
    for col in use_mode:
        th_data[col] = th_data[col].fillna(th_data[col].mode()[0])
    return th_data

# --------------------------
# A9: Normalization
# --------------------------
def normalize_data(th_data):
    numeric_cols = th_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    minmax = th_data.copy()
    zscore = th_data.copy()
    for col in numeric_cols:
        minmax[col] = (th_data[col] - th_data[col].min()) / (th_data[col].max() - th_data[col].min())
        zscore[col] = (th_data[col] - th_data[col].mean()) / th_data[col].std()
    return minmax, zscore

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    excel = r"C:/Users/bramj/OneDrive/Desktop/ML_Lab2/Lab Session Data.xlsx"

    # --- A1 ---
    purchase_data = load_excel_data(excel, 'Purchase data')
    A, C, dim, num_vecs, rank, unit_costs = analyze_purchase_data(purchase_data)
    print("A1:", dim, num_vecs, rank, unit_costs)

    # --- A2 ---
    labels = label_rich_poor(purchase_data)
    model, acc = train_rich_poor_classifier(A, labels)
    print("A2 Accuracy:", acc)

    # --- A3 ---
    stock_data = load_excel_data(excel, 'IRCTC Stock Price')
    mean_price, var_price = stock_statistics(stock_data)
    mean_wed = mean_by_day(stock_data, 'Wed')
    mean_apr = mean_by_month(stock_data, 'Apr')
    p_loss = probability_of_loss(stock_data)
    p_profit_wed = probability_profit_wednesday(stock_data)
    p_cond_profit_wed = conditional_profit_given_wednesday(stock_data)

    print("\nA3:")
    print("Population Mean:", mean_price, "Variance:", var_price)
    print("Mean (Wed):", mean_wed, "Mean (Apr):", mean_apr)
    print("P(Loss):", p_loss)
    print("P(Profit on Wed):", p_profit_wed)
    print("P(Profit | Wed):", p_cond_profit_wed)
    plot_chg_vs_day(stock_data)

    # --- A4 ---
    thyroid_data = load_excel_data(excel, 'thyroid0387_UCI')
    num_cols, cat_cols, missing = thyroid_exploration(thyroid_data)
    print("\nA4:", num_cols, cat_cols, missing.sum())

    # --- A5 ---
    B1 = [1 if str(x).lower() == 't' else 0 for x in thyroid_data['on thyroxine']]
    B2 = [1 if str(x).lower() == 't' else 0 for x in thyroid_data['query on thyroxine']]
    print("\nA5 JC:", jaccard(B1, B2), "SMC:", smc(B1, B2))

    # --- A6 ---
    print("A6 Cosine Similarity:", cosine_similarity(B1, B2))

    # --- A7 ---
    compute_heatmaps(thyroid_data, n=15)

    # --- A8 ---
    thyroid_data.replace("?", pd.NA, inplace=True)
    imputed = impute_data(thyroid_data, ['T3'], ['TSH', 'TT4'], ['sex'])
    print("A8 Missing After Imputation:", imputed.isna().sum().sum())

    # --- A9 ---
    minmax, zscore = normalize_data(imputed)
    print("A9 MinMax (head):\n", minmax.head())
