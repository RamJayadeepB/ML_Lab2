import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [20, 6, 2],
    [16, 3, 6],
    [27, 6, 2],
    [19, 1, 2],
    [24, 4, 2],
    [22, 1, 5],
    [15, 4, 2],
    [18, 4, 2],
    [21, 1, 4],
    [16, 2, 4]
])

payments = np.array([386, 289, 393, 110, 280, 167, 271, 274, 148, 198])
y = np.where(payments > 200, 1, 0)  # 1 = RICH, 0 = POOR

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

all_preds = clf.predict(X)
for i, label in enumerate(all_preds):
    status = "RICH" if label == 1 else "POOR"
    print(f"Customer C_{i+1} is classified as: {status}")
