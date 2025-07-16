import numpy as np

A = np.array([
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

C = np.array([386, 289, 393, 110, 280, 167, 271, 274, 148, 198])

rank_A = np.linalg.matrix_rank(A)

A_pinv = np.linalg.pinv(A)

X = A_pinv.dot(C)

print("Rank of A:", rank_A)
print("Cost Vector X (Prices):")
print("Candy price: Rs", round(X[0], 2))
print("Mango price: Rs", round(X[1], 2))
print("Milk Packet price: Rs", round(X[2], 2))
