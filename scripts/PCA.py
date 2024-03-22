import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen

np.random.seed(42)


def generate_2D_gaussian_distribution() -> multivariate_normal_frozen:
    """
    二次元正規分布を生成してプロットする関数

    Returns
    -------
    rv : multivariate_normal_frozen
        二次元正規分布

    """
    # 二次元正規分布の平均と共分散行列を定義
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
