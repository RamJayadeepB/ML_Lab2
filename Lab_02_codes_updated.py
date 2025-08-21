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
    A = data.iloc[:, 1:4]
    C = data.iloc[:, 4:5]
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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
    return st.mean(data['Price'][data['Day'] == day])

def mean_by_month(data, month):
    return st.mean(data['Price'][data['Month'] == month])

def probability_of_loss(data):
    loss = data['Chg%'].apply(lambda x: 1 if x < 0 else 0)
    return sum(loss)/len(loss)

def probability_profit_wednesday(data):
    wed_data = data[data['Day'] == 'Wed']
    return (wed_data['Chg%'] > 0).mean()

def conditional_profit_given_wednesday(data):
    wed_data = data[data['Day'] == 'Wed']
    if len(wed_data) == 0:
        return np.nan
    return (wed_data['Chg%'] > 0).sum() / len(wed_data)

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
    dot_product = sum(a*b for a, b in zip(v1, v2))
    norm1 = np.sqrt(sum(a**2 for a in v1))
    norm2 = np.sqrt(sum(b**2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

# --------------------------
# A7: Heatmaps
# --------------------------
def compute_heatmaps(df, n=20):
    # Normalize all string values
    df = df.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)

    # Replace t/f with 1/0, handle missing
    binary_df = (
        df.replace({'t': 1, 'f': 0, '?': np.nan})
          .apply(pd.to_numeric, errors='coerce')
          .fillna(0)
          .astype(int)
          .iloc[:n]
    )

    # Compute similarity matrices
    JC  = pd.DataFrame([[jaccard(binary_df.iloc[i], binary_df.iloc[j]) for j in range(n)] for i in range(n)])
    SMC = pd.DataFrame([[smc(binary_df.iloc[i], binary_df.iloc[j])     for j in range(n)] for i in range(n)])
    COS = pd.DataFrame([[cosine_similarity(binary_df.iloc[i], binary_df.iloc[j]) for j in range(n)] for i in range(n)])

    # Plot heatmaps
    plt.figure(figsize=(30, 8))

    plt.subplot(1, 3, 1)
    sns.heatmap(JC, annot=False, cmap='Blues')
    plt.title('Jaccard Coefficient (JC)')

    plt.subplot(1, 3, 2)
    sns.heatmap(SMC, annot=False, cmap='Greens')
    plt.title('Simple Matching Coefficient (SMC)')

    plt.subplot(1, 3, 3)
    sns.heatmap(COS, annot=False, cmap='Oranges')
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
# Main Section
# --------------------------
if __name__ == "__main__":
    excel = "C:/Users/bramj/OneDrive/Desktop/ML_Lab2/Lab Session Data.xlsx"

    # A1
    purchase_data = load_excel_data(excel, 'Purchase data')
    A, C, dim, num_vecs, rank, unit_costs = analyze_purchase_data(purchase_data)
    print("A1 Results:", dim, num_vecs, rank, unit_costs)

    # A2
    labels = label_rich_poor(purchase_data)
    model, acc = train_rich_poor_classifier(A, labels)
    print("A2 Accuracy:", acc)

    # A3
    stock_data = load_excel_data(excel, 'IRCTC Stock Price')
    mean_price, var_price = stock_statistics(stock_data)
    print("A3 Mean, Var:", mean_price, var_price)
    print("Loss probability:", probability_of_loss(stock_data))

    # A4
    thyroid_data = load_excel_data(excel, 'thyroid0387_UCI')
    num_cols, cat_cols, missing = thyroid_exploration(thyroid_data)
    print("A4 Exploration:", num_cols, cat_cols, missing.sum())

    # A5
    B1 = [1 if str(x).lower() == 't' else 0 for x in thyroid_data['on thyroxine']]
    B2 = [1 if str(x).lower() == 't' else 0 for x in thyroid_data['query on thyroxine']]
    jc = jaccard(B1, B2)
    smc_val = smc(B1, B2)
    print("A5 JC:", jc, "SMC:", smc_val)

    # A6
    cos = cosine_similarity(B1, B2)
    print("A6 Cosine Similarity:", cos)

    # A7
    JC, SMC, COS = compute_heatmaps(thyroid_data, n=15)

    # A8
    thyroid_data.replace("?", pd.NA, inplace=True)
    imputed = impute_data(thyroid_data, ['T3'], ['TSH', 'TT4'], ['sex'])
    print("A8 Imputation Done. Missing:", imputed.isna().sum().sum())

    # A9
    minmax, zscore = normalize_data(imputed)
    print("A9 Normalization Done. MinMax head:\n", minmax.head())
