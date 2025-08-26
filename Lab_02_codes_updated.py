import pandas as pd
import numpy as np
import statistics as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# Helper Function: Load Data from Excel
# =====================================================
def load_excel_data(excel, sheet):
    return pd.read_excel(excel, sheet_name=sheet)

# =====================================================
# A1: Linear Algebra on Purchase Data
# =====================================================
def analyze_purchase_data(data):
    """
    Performs linear algebra analysis on purchase data:
    - Extracts feature matrix (A) and target vector (C)
    """
    A = data.iloc[:, 1:4]  # Features (columns 1–3)
    C = data.iloc[:, 4:5]  # Target (column 4)
    dim = A.shape[1]       # Number of features (columns)
    num_vectors = A.shape[0]  # Number of samples (rows)
    rank = np.linalg.matrix_rank(A)  # Rank = independent columns in A
    unit_costs = np.dot(np.linalg.pinv(A), C)  # Pseudo-inverse to solve Ax ≈ C
    return A, C, dim, num_vectors, rank, unit_costs

# =====================================================
# A2: Classification of Customers (RICH/POOR)
# =====================================================
def label_rich_poor(data, threshold=200):
    """Labels customers as RICH or POOR based on payment threshold."""
    return np.where(data['Payment (Rs)'] > threshold, "RICH", "POOR")

def train_rich_poor_classifier(X, y):
    """
    Trains a K-Nearest Neighbors (KNN) classifier:
    - Splits data into training/testing sets
    - Fits model on training data
    - Returns trained model and test accuracy
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=3)  # KNN with 3 neighbors
    model.fit(X_train, y_train)  # Train model
    acc = accuracy_score(y_test, model.predict(X_test))  # Test accuracy
    return model, acc

# =====================================================
# A3: IRCTC Stock Price Analysis (Statistics & Probability)
# =====================================================
def stock_statistics(data):
    """Calculates population mean and variance of stock prices."""
    mean_price = st.mean(data['Price'])
    var_price = st.variance(data['Price'])
    return mean_price, var_price

def mean_by_day(data, day):
    """Calculates mean price for a given day of the week."""
    return st.mean(data['Price'][data['Day'] == day])

def mean_by_month(data, month):
    """Calculates mean price for a given month."""
    return st.mean(data['Price'][data['Month'] == month])

def probability_of_loss(data):
    """Probability that stock experienced a loss (<0% change)."""
    loss = data['Chg%'].apply(lambda x: 1 if x < 0 else 0)
    return sum(loss) / len(loss)

def probability_profit_wednesday(data):
    """Probability of profit specifically on Wednesdays."""
    wed_data = data[data['Day'] == 'Wed']
    return (wed_data['Chg%'] > 0).mean()

def conditional_profit_given_wednesday(data):
    """Conditional probability of profit given it's Wednesday."""
    wed_data = data[data['Day'] == 'Wed']
    return (wed_data['Chg%'] > 0).sum() / len(wed_data) if len(wed_data) > 0 else np.nan

def stock_analysis(data):
    """Performs full stock analysis and visualizes Chg% by day."""
    print("\n===== A3: IRCTC Stock Analysis =====")
    mean_price, var_price = stock_statistics(data)
    print(f"Population Mean Price: {mean_price}, Variance: {var_price}")
    print(f"Wednesday Sample Mean: {mean_by_day(data, 'Wed')}")
    print(f"April Sample Mean: {mean_by_month(data, 'Apr')}")
    print(f"Probability of Loss: {probability_of_loss(data)}")
    print(f"Probability of Profit on Wednesday: {probability_profit_wednesday(data)}")
    print(f"Conditional Probability(Profit|Wednesday): {conditional_profit_given_wednesday(data)}")

    # Scatterplot of % change vs Day of week
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=data['Day'], y=data['Chg%'])
    plt.title('Chg% vs Day of Week')
    plt.show()

# =====================================================
# A4: Thyroid Data Exploration (EDA)
# =====================================================
def thyroid_exploration(th_data):
    """
    Explores Thyroid dataset:
    - Identifies numeric & categorical columns
    - Suggests encoding (Label vs One-Hot)
    - Detects missing values & outliers (IQR)
    - Prints summary statistics
    """
    print("\n===== A4: Data Exploration =====")

    numeric_cols = th_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = th_data.select_dtypes(include=['object', 'bool']).columns.tolist()
    print(f"Numeric Columns: {numeric_cols}")
    print(f"Categorical Columns: {categorical_cols}")

    # Encoding suggestions
    print("\nColumn Types & Suggested Encoding:")
    for col in th_data.columns:
        unique_vals = th_data[col].dropna().unique()
        if len(unique_vals) == 2:
            print(f"{col}: Binary -> Label Encoding")
        elif col in numeric_cols:
            print(f"{col}: Continuous (Numeric)")
        else:
            print(f"{col}: Nominal -> One-Hot Encoding")

    # Range of numeric values
    print("\nNumeric Column Ranges:")
    for col in numeric_cols:
        print(f"{col}: Min={th_data[col].min()}, Max={th_data[col].max()}")

    # Missing values
    print("\nMissing Values:")
    print(th_data.isnull().sum())

    # Outliers detection using IQR
    print("\nOutliers (IQR Method):")
    for col in numeric_cols:
        Q1, Q3 = th_data[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = ((th_data[col] < (Q1 - 1.5 * IQR)) | (th_data[col] > (Q3 + 1.5 * IQR))).sum()
        print(f"{col}: {outliers} outliers")

    # Mean and Std deviation of numeric columns
    print("\nMean & Std Dev of Numeric Columns:")
    for col in numeric_cols:
        print(f"{col}: Mean={th_data[col].mean()}, Std={th_data[col].std()}")

    return numeric_cols, categorical_cols

# =====================================================
# A5: Similarity Measures (Jaccard & SMC)
# =====================================================
def jaccard(B1, B2):
    """Computes Jaccard Coefficient for two binary vectors."""
    f11 = sum(a == 1 and b == 1 for a, b in zip(B1, B2))
    f10 = sum(a == 1 and b == 0 for a, b in zip(B1, B2))
    f01 = sum(a == 0 and b == 1 for a, b in zip(B1, B2))
    return f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else np.nan

def smc(B1, B2):
    """Computes Simple Matching Coefficient (SMC) for two binary vectors."""
    f11 = sum(a == 1 and b == 1 for a, b in zip(B1, B2))
    f00 = sum(a == 0 and b == 0 for a, b in zip(B1, B2))
    f10 = sum(a == 1 and b == 0 for a, b in zip(B1, B2))
    f01 = sum(a == 0 and b == 1 for a, b in zip(B1, B2))
    return (f11 + f00) / (f11 + f00 + f10 + f01)

def binary_similarity(th_data):
    """
    Converts binary-like columns to 0/1 and computes
    Jaccard & SMC similarity for first 2 records.
    """
    print("\n===== A5: Jaccard & SMC =====")
    binary_cols = [col for col in th_data.columns if th_data[col].nunique() == 2]
    binary_df = th_data[binary_cols].applymap(lambda x: 1 if str(x).lower() in ['t', 'yes', 'y', '1'] else 0)
    v1, v2 = binary_df.iloc[0], binary_df.iloc[1]
    print(f"Jaccard Coefficient: {jaccard(v1, v2)}")
    print(f"Simple Matching Coefficient: {smc(v1, v2)}")
    return binary_df

# =====================================================
# A6: Cosine Similarity
# =====================================================
def cosine_similarity(v1, v2):
    """Computes cosine similarity between two numeric vectors."""
    dot_product = np.dot(v1, v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

def cosine_on_full_vectors(th_data):
    """
    Converts categorical to numeric, handles missing values,
    and computes cosine similarity for first 2 records.
    """
    print("\n===== A6: Cosine Similarity =====")
    df_numeric = th_data.applymap(lambda x: 1 if str(x).lower() in ['t', 'yes', 'y', '1'] else x)
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)
    v1, v2 = df_numeric.iloc[0], df_numeric.iloc[1]
    print(f"Cosine Similarity: {cosine_similarity(v1, v2)}")
    return df_numeric

# =====================================================
# A7: Heatmap of Pairwise Similarities
# =====================================================
def heatmap_similarity(df_numeric, n=20):
    """
    Computes pairwise Jaccard, SMC, and Cosine similarities
    for first n records and visualizes them as heatmaps.
    """
    print("\n===== A7: Heatmap Similarities =====")
    df20 = df_numeric.iloc[:n]
    JC = pd.DataFrame([[jaccard(df20.iloc[i], df20.iloc[j]) for j in range(n)] for i in range(n)])
    SMC = pd.DataFrame([[smc(df20.iloc[i], df20.iloc[j]) for j in range(n)] for i in range(n)])
    COS = pd.DataFrame([[cosine_similarity(df20.iloc[i], df20.iloc[j]) for j in range(n)] for i in range(n)])

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.heatmap(JC, cmap='Blues')
    plt.title("Jaccard Coefficient")

    plt.subplot(1, 3, 2)
    sns.heatmap(SMC, cmap='Greens')
    plt.title("Simple Matching Coefficient")

    plt.subplot(1, 3, 3)
    sns.heatmap(COS, cmap='Oranges')
    plt.title("Cosine Similarity")
    plt.tight_layout()
    plt.show()

# =====================================================
# A8: Data Imputation
# =====================================================
def impute_data(df, use_mean, use_median, use_mode):
    """
    Replaces missing values:
    - Continuous columns with mean/median
    - Categorical columns with mode
    """
    print("\n===== A8: Data Imputation =====")
    for col in use_mean:
        df[col] = df[col].fillna(df[col].mean())
    for col in use_median:
        df[col] = df[col].fillna(df[col].median())
    for col in use_mode:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("Missing after imputation:", df.isnull().sum().sum())
    return df

# =====================================================
# A9: Data Normalization
# =====================================================
def normalize_data(df):
    """
    Applies:
    - Min-Max Scaling (0–1)
    - Z-Score Normalization (mean=0, std=1)
    """
    print("\n===== A9: Data Normalization =====")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    minmax = df.copy()
    zscore = df.copy()
    for col in numeric_cols:
        minmax[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        zscore[col] = (df[col] - df[col].mean()) / df[col].std()
    print("Normalization Done.")
    return minmax, zscore

# =====================================================
# MAIN PROGRAM EXECUTION
# =====================================================
if __name__ == "__main__":
    excel = r"C:\Users\DELL\Downloads\Lab Session Data (1).xlsx"

    # A1 & A2: Linear Algebra + KNN Classification
    purchase_data = load_excel_data(excel, 'Purchase data')
    A, C, dim, num_vecs, rank, unit_costs = analyze_purchase_data(purchase_data)
    print("\n===== A1 =====")
    print(f"Dimensionality: {dim}, Vectors: {num_vecs}, Rank: {rank}, Unit Costs: {unit_costs.T}")

    labels = label_rich_poor(purchase_data)
    model, acc = train_rich_poor_classifier(A, labels)
    print("\n===== A2 =====")
    print(f"Classifier Accuracy: {acc}")

    # A3: Stock Price Analysis
    stock_data = load_excel_data(excel, 'IRCTC Stock Price')
    stock_analysis(stock_data)

    # A4: Thyroid Data Exploration
    thyroid_data = load_excel_data(excel, 'thyroid0387_UCI')
    numeric_cols, cat_cols = thyroid_exploration(thyroid_data)

    # A5 & A6: Similarity Measures
    binary_df = binary_similarity(thyroid_data)
    df_numeric = cosine_on_full_vectors(thyroid_data)

    # A7: Heatmap of Similarities
    heatmap_similarity(df_numeric, n=20)

    # A8: Data Imputation
    thyroid_data.replace("?", np.nan, inplace=True)
    imputed = impute_data(thyroid_data, use_mean=['T3'], use_median=['TSH', 'TT4'], use_mode=['sex'])

    # A9: Normalization
    minmax, zscore = normalize_data(imputed)
    print("MinMax Example:\n", minmax.head())
    print("Z-Score Example:\n", zscore.head())
