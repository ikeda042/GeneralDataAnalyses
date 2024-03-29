import pandas as pd
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns

headers = ["rpoS", "dnaK", "oxyR", "sosR", "cspA"]

data1 = [
    [0.61, 0.16, 0.08, 0.05, 0.10],
    [0.54, 0.18, 0.07, 0.10, 0.12],
    [0.57, 0.10, 0.06, 0.15, 0.12],
    [0.39, 0.16, 0.17, 0.13, 0.15],
    [0.45, 0.14, 0.18, 0.14, 0.08],
    [0.42, 0.08, 0.12, 0.19, 0.19],
    [0.50, 0.17, 0.05, 0.16, 0.11],
    [0.44, 0.16, 0.08, 0.18, 0.15],
    [0.51, 0.12, 0.23, 0.05, 0.09],
    [0.59, 0.04, 0.12, 0.05, 0.20],
]
data2 = [
    [0.19, 0.60, 0.01, 0.08, 0.12],
    [0.19, 0.44, 0.12, 0.15, 0.11],
    [0.14, 0.46, 0.18, 0.11, 0.11],
    [0.11, 0.61, 0.08, 0.14, 0.07],
    [0.05, 0.53, 0.15, 0.16, 0.12],
    [0.02, 0.50, 0.16, 0.16, 0.16],
    [0.16, 0.54, 0.10, 0.07, 0.13],
    [0.04, 0.53, 0.16, 0.17, 0.10],
    [0.17, 0.46, 0.08, 0.12, 0.17],
    [0.19, 0.57, 0.02, 0.12, 0.10],
]

df1 = pd.DataFrame(data1, columns=headers)
df2 = pd.DataFrame(data2, columns=headers)

print(df1)
print(df2)

A = np.concatenate([data1, data2], axis=0)
U, S, V_T = svd(A)
print("左特異値ベクトル行列")
print(U)
print("Σ")
print(np.diag(S))
print("右特異値ベクトル行列")
print(V_T)

A = np.concatenate([data1, data2], axis=0)

# Performing Singular Value Decomposition (SVD)
U, S, V_T = svd(A)

print(S)

fig = plt.figure(figsize=(9, 6))
sns.set()
plt.plot([i for i in range(1, 6)], S, color="blue", linewidth=1, marker="o")
plt.xlabel("Dimention(-)")
plt.ylabel("Singular value(-)")
plt.savefig("images/swiss_roll_SVD.png", dpi=500)


reduced_data = np.dot(A, V_T.T[:, :3])

# Separating the projected data for the two original datasets
reduced_data1 = reduced_data[: len(data1)]
reduced_data2 = reduced_data[len(data1) :]


plt.figure(figsize=(9, 6))
ax = plt.axes(projection="3d")
ax.view_init(elev=20, azim=20)
ax.scatter3D(
    reduced_data1[:, 0],
    reduced_data1[:, 1],
    reduced_data1[:, 2],
    color="blue",
    label="Negative Ctrl.",
    alpha=0.6,
)
ax.scatter3D(
    reduced_data2[:, 0],
    reduced_data2[:, 1],
    reduced_data2[:, 2],
    color="red",
    label="Positive Ctrl.",
    alpha=0.6,
)
plt.title("3D Projection using SVD")
plt.xlabel(f"Component 1 s = {round(S[0],1)}")
plt.ylabel(f"Component 2 s = {round(S[1],1)} ")
plt.legend()
plt.savefig("result_3D.png", dpi=500)
