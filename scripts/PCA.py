import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from numpy.linalg import eig
import math
import pandas as pd

np.random.seed(42)


def generate_2D_gaussian_distribution() -> multivariate_normal_frozen:
    """
    二次元正規分布を生成してプロットする関数

    Returns
    -------
    rv : multivariate_normal_frozen
        二次元正規分布

    """
    # 二次元正規分布の平均と分散共分散行列を定義
    mean = [0, 0]
    covariance = np.array([[1, 0.5], [0.5, 1]])

    # 二次元正規分布を生成
    rv = multivariate_normal(mean, covariance)

    # プロットする範囲を指定
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # 二次元正規分布の確率密度関数を計算
    Z = rv.pdf(pos)

    # プロット
    fig = plt.figure(figsize=(9, 6))
    plt.contourf(X, Y, Z, levels=20, cmap="inferno")
    plt.colorbar(label="Probability Density")
    plt.xlabel("X")
    plt.ylabel("Y")
    fig.savefig("images/result_PCA.png", dpi=500)

    return rv


# プロットする点の数
num_points = 1000

# 乱数を生成して二次元正規分布からサンプリング
samples = generate_2D_gaussian_distribution().rvs(num_points)

# サンプルした点のx座標とy座標を取得
x_samples = samples[:, 0]
y_samples = samples[:, 1]

# プロット
fig = plt.figure()
plt.scatter(x_samples, y_samples, alpha=0.4, color="black", s=24)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
fig.savefig("images/result_PCA_sampled_points.png", dpi=500)

##########################################################################################################################################


X = np.array([x_samples, y_samples])

# サンプルデータの分散共分散行列を計算
Cov = np.cov(X)

# 分散共分散行列の固有値と固有ベクトルを計算
eigenvalues, eigenvectors = eig(Cov)


# 最大固有値に対応する固有ベクトルを取得
max_index = np.argmax(eigenvalues)

# 主成分分析の結果を表示
m = eigenvectors[0][max_index] / eigenvectors[1][max_index]
k_t = 0
PC1 = eigenvalues[max_index] / sum(eigenvalues)
PC2 = eigenvalues[1 - max_index] / sum(eigenvalues)
Q = np.array(eigenvectors[1 - max_index])
w = eigenvectors[max_index]


# 元のプロットに第一主成分を重ね合わせる
fig = plt.figure()
x = np.linspace(-5, 5, 1000)
plt.scatter(x_samples, y_samples, alpha=0.4, color="black", s=14)
plt.scatter(x, [i * m + k_t for i in x], s=1, color="red")
plt.xlabel(f"X")
plt.ylabel("Y")
plt.title(f"PC1: {math.floor((PC1*1000))/10}%")
plt.grid(True)
fig.savefig("images/result_PCA_PC1.png", dpi=500)
plt.close()

# 射影
U = X.transpose() @ np.array(eigenvectors[0])
df = pd.DataFrame(
    {
        "U_PC1": U,
    }
)
df_melt = pd.melt(df)
print(df_melt.head())
