from __future__ import annotations
import cv2
import numpy as np
import pickle
from numpy.linalg import inv
from sqlalchemy import create_engine, Column, Integer, String, BLOB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import cv2
import matplotlib.pyplot as plt

plt.style.use("dark_background")
from scipy.optimize import minimize
from scipy.linalg import eig
import os
from dataclasses import dataclass
import shutil
import scipy.integrate


@dataclass
class Point:
    def __init__(self, u1: float, G: float):
        self.u1 = u1
        self.G = G

    def __gt__(self, other):
        return self.u1 > other.u1

    def __lt__(self, other):
        return self.u1 < other.u1

    def __repr__(self) -> str:
        return f"({self.u1},{self.G})"


class Cells:
    def __init__(
        self, db_path: str = "database_sk326.db", only_ph: bool = False
    ) -> None:
        Base = declarative_base()

        class Cell_(Base):
            __tablename__ = "cells"
            id = Column(Integer, primary_key=True)
            cell_id = Column(String)
            img_ph = Column(BLOB)
            img_fluo1 = Column(BLOB, nullable=True)
            contour = Column(BLOB)

        engine: create_engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session: sessionmaker = sessionmaker(bind=engine)
        session = Session()
        cells: list[Cell_] = session.query(Cell_).all()
        print("Loading cells from database...")
        cells.sort(key=lambda c: c.cell_id)
        self.cells: list[Cell] = [
            Cell(
                cell.cell_id,
                image_ph=cv2.imdecode(
                    np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_COLOR
                ),
                image_fluo=(
                    cv2.imdecode(
                        np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if cell.img_fluo1 is not None
                    else None
                ),
                contour=cell.contour,
                only_ph=only_ph,
            )
            for cell in cells
        ]

    def get_cell(self, cell_id: str) -> Cell | None:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None

    def get_cells(self) -> list[Cell]:
        return self.cells

    def __repr__(self) -> str:
        ret: str = ""
        count_info: str = (
            f"\n++++++++++++++++++++++\nTotal cells -> {str(len(self.cells))}"
        )
        for cell in self.cells:
            ret += f"{cell.cell_id}\n"
        ret += count_info
        return ret


class Cell:
    def __init__(
        self,
        cell_id: str,
        image_ph: np.ndarray,
        image_fluo: np.ndarray,
        contour: np.ndarray,
        only_ph: bool,
    ) -> None:
        self.cell_id: str = cell_id
        self.image_ph: np.ndarray = image_ph
        self.contour: np.ndarray = contour
        if not only_ph:
            self.image_fluo: np.ndarray = image_fluo
            self.image_fluo_gray: np.ndarray = cv2.cvtColor(
                self.image_fluo, cv2.COLOR_BGR2GRAY
            )
        # make directories
        self.dir_base: str = "images"
        self.dir_ph: str = f"{self.dir_base}/ph"
        self.dir_fluo: str = f"{self.dir_base}/fluo"
        self.dir_replot: str = f"{self.dir_base}/replot"
        self.dir_path: str = f"{self.dir_base}/path"
        self.dir_volume: str = f"{self.dir_base}/volume"
        self.dir_cylinders: str = f"{self.dir_base}/cylinders"
        if not os.path.exists(self.dir_base):
            os.makedirs(self.dir_base)
        else:
            shutil.rmtree(self.dir_base)
            os.makedirs(self.dir_base)
        if not os.path.exists(self.dir_ph):
            os.makedirs(self.dir_ph)
        if not os.path.exists(self.dir_fluo):
            os.makedirs(self.dir_fluo)
        if not os.path.exists(self.dir_replot):
            os.makedirs(self.dir_replot)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        if not os.path.exists(self.dir_volume):
            os.makedirs(self.dir_volume)
        if not os.path.exists(self.dir_cylinders):
            os.makedirs(self.dir_cylinders)

    def __repr__(self) -> str:
        return f"Cell ID: {self.cell_id}"

    def write_image(self, only_ph: bool, out_dir: str | None = None) -> None:
        if out_dir is None:
            out_name_ph: str = f"{self.dir_ph}/{self.cell_id}_ph.png"
            out_name_fluo: str = f"{self.dir_fluo}/{self.cell_id}_fluo.png"
        else:
            if not os.path.exists(f"{out_dir}/ph"):
                os.makedirs(f"{out_dir}/ph")
            if not os.path.exists(f"{out_dir}/fluo"):
                os.makedirs(f"{out_dir}/fluo")
            out_name_ph: str = f"{out_dir}/ph/{self.cell_id}_ph.png"
            out_name_fluo: str = f"{out_dir}/fluo/{self.cell_id}_fluo.png"
        image_ph_copy: np.ndarray = self.image_ph.copy()
        cv2.drawContours(image_ph_copy, pickle.loads(self.contour), -1, (0, 255, 0), 1)
        if not only_ph:
            image_fluo_copy: np.ndarray = self.image_fluo.copy()
            cv2.drawContours(
                image_fluo_copy, pickle.loads(self.contour), -1, (0, 255, 0), 1
            )
            cv2.imwrite(out_name_fluo, image_fluo_copy)
        cv2.imwrite(out_name_ph, image_ph_copy)
        # print(f"Images written to {out_name_ph} and {out_name_fluo}")

    def get_contour(self) -> np.ndarray:
        return self.contour

    def get_image_ph(self) -> np.ndarray:
        return self.image_ph

    def get_image_fluo(self) -> np.ndarray:
        return self.image_fluo

    def replot_contour(self, polyfit_degree: int | None) -> np.ndarray:
        contour = [list(i[0]) for i in pickle.loads(self.contour)]

        X = np.array(
            [
                [i[1] for i in contour],
                [i[0] for i in contour],
            ]
        )

        # 基底変換関数を呼び出して必要な変数を取得
        (
            u1,
            u2,
            u1_contour,
            u2_contour,
            min_u1,
            max_u1,
            u1_c,
            u2_c,
            U,
            contour_U,
        ) = self._basis_conversion(
            contour,
            X,
            self.image_ph.shape[0] / 2,
            self.image_ph.shape[1] / 2,
            contour,
        )

        if polyfit_degree is None or polyfit_degree == 1:

            # 中心座標(u1_c, u2_c)が(0,0)になるように補正
            u1_adj = u1 - u1_c
            u2_adj = u2 - u2_c

            # 細胞を長軸ベースに細分化(Meta parameters)
            split_num = 20
            deltaL = cell_length / split_num

            fig = plt.figure(figsize=(6, 6))
            plt.scatter(u1_adj, u2_adj, s=5, color="lime")
            plt.scatter(0, 0, color="red", s=100)
            plt.axis("equal")
            margin_width = 10
            margin_height = 10
            plt.xlim([min(u1_adj) - margin_width, max(u1_adj) + margin_width])
            plt.ylim([min(u2_adj) - margin_height, max(u2_adj) + margin_height])

            x = np.linspace(min(u1_adj), max(u1_adj), 1000)
            theta = self._poly_fit(np.array([u1_adj, u2_adj]).T)
            y = np.polyval(theta, x)
            plt.plot(x, y, color="red")
            plt.xlabel("u1")
            plt.ylabel("u2")
            plt.savefig(f"{self.dir_replot}/{self.cell_id}_replot.png")
            plt.savefig("realtime_replot.png")
            plt.close(fig)

            # 細胞情報の初期化および定義
            cell_length = max(u1_adj) - min(u1_adj)
            area = cv2.contourArea(np.array(contour))
            volume = 0

            # u_2をすべて正にする
            fig_volume = plt.figure(figsize=(6, 6))
            u_2_abs = [abs(i) for i in u2_adj]

            plt.scatter(u1_adj, u_2_abs, s=5, color="lime")
            # plt.scatter(0, 0, color="red", s=100)
            plt.axis("equal")

            margin_width = 10
            margin_height = 10
            plt.xlim([min(u1_adj) - margin_width, max(u1_adj) + margin_width])
            plt.ylim([min(u_2_abs) - margin_height, max(u_2_abs) + margin_height])
            y = np.polyval(theta, x)

            # 区間ΔLごとに分割して、縦の線を引く。この時、縦の線のy座標はその線に最も近い点のy座標とする。
            points_init = [
                p
                for p in [[i, j] for i, j in zip(u1_adj, u_2_abs)]
                if min(u1_adj) <= p[0] <= min(u1_adj) + deltaL
            ]

            # 区間中のyの平均値を求める
            y_mean = sum([i[1] for i in points_init]) / len(points_init)
            plt.scatter(
                (min(u1_adj) + min(u1_adj + deltaL)) / 2, y_mean, color="magenta", s=20
            )
            plt.plot([min(u1_adj), min(u1_adj)], [0, y_mean], color="lime")

            # 円柱ポリゴンの定義
            cylinders = []

            # 幅を格納
            widths = []

            for i in range(0, split_num):
                x_0 = min(u1_adj) + i * deltaL
                x_1 = min(u1_adj) + (i + 1) * deltaL
                points = [
                    p
                    for p in [[i, j] for i, j in zip(u1_adj, u_2_abs)]
                    if x_0 <= p[0] <= x_1
                ]
                if len(points) == 0:
                    # 前の点を使う
                    y_mean = y_mean
                else:
                    # 区間中のyの平均値を求める
                    y_mean = sum([i[1] for i in points]) / len(points)
                plt.scatter(((x_0) + (x_1)) / 2, y_mean, color="magenta", s=20)
                plt.plot([x_0, x_0], [0, y_mean], color="lime")

                volume += y_mean**2 * np.pi * deltaL

                cylinders.append((x_0, deltaL, y_mean, "lime", 0.3))

                widths.append(y_mean)

            plt.xlabel("u1")
            plt.ylabel("u2")
            plt.savefig(f"{self.dir_volume}/{self.cell_id}_volume.png", dpi=300)
            plt.close(fig_volume)
            Cell._plot_cylinders(
                cylinders, f"{self.dir_cylinders}/{self.cell_id}_cylinders.png"
            )

            # width はwidthsの大きい順から3つの平均値を取る。
            # widthsの各値は、その区間のy座標の平均値である。
            # この際、区間のy軸方向は細胞の片側の幅を表すため、値を単純に二倍する。
            widths = sorted(widths, reverse=True)
            widths = widths[:3]
            width = sum(widths) / len(widths)
            width *= 2
            return (area, volume, width, cell_length)
        else:

            fig = plt.figure(figsize=(6, 6))
            plt.scatter(u1, u2, s=5, color="lime")
            plt.scatter(u1_c, u2_c, color="red", s=100)
            plt.axis("equal")
            margin_width = 10
            margin_height = 10
            plt.xlim([min(u1) - margin_width, max(u1) + margin_width])
            plt.ylim([min(u2) - margin_height, max(u2) + margin_height])

            x = np.linspace(min(u1), max(u1), 100)

            # 多項式フィッティングの際にu1,u2を入れ替える事に注意
            theta = self._poly_fit(np.array([u2, u1]).T, degree=polyfit_degree)
            y = np.polyval(theta, x)
            plt.plot(x, y, color="red")
            plt.xlabel("u1")
            plt.ylabel("u2")
            plt.savefig(f"{self.dir_replot}/{self.cell_id}_replot.png")
            plt.savefig("realtime_replot.png")
            plt.close(fig)

            cell_length = Cell._calc_arc_length(theta, min(u1), max(u1))
            area = cv2.contourArea(np.array(contour))
            volume = 0

            # 細胞を長軸ベースに細分化(Meta parameters)
            split_num = 20
            deltaL = cell_length / split_num

            raw_points: list[list[float]] = []
            for i, j in zip(u1, u2):
                min_distance, min_point = Cell._find_minimum_distance_and_point(
                    theta, i, j
                )
                arc_length = Cell._calc_arc_length(theta, min(u1), i)
                raw_points.append([arc_length, min_distance])

            # 区間ΔLごとに分割して、縦の線を引く。この時、縦の線のy座標はその線に最も近い点のy座標とする。
            points_init = [
                p
                for p in [[i, j] for i, j in zip(u1_adj, u_2_abs)]
                if min(u1_adj) <= p[0] <= min(u1_adj) + deltaL
            ]

            # 区間中のyの平均値を求める
            y_mean = sum([i[1] for i in points_init]) / len(points_init)

            # 円柱ポリゴンの定義
            cylinders = []

            # 幅を格納
            widths = []

            for i in range(0, split_num):
                x_0 = min(u1_adj) + i * deltaL
                x_1 = min(u1_adj) + (i + 1) * deltaL
                points = [
                    p
                    for p in [[i, j] for i, j in zip(u1_adj, u_2_abs)]
                    if x_0 <= p[0] <= x_1
                ]
                if len(points) == 0:
                    # 前の点を使う
                    y_mean = y_mean
                else:
                    # 区間中のyの平均値を求める
                    y_mean = sum([i[1] for i in points]) / len(points)
                plt.scatter(((x_0) + (x_1)) / 2, y_mean, color="magenta", s=20)
                plt.plot([x_0, x_0], [0, y_mean], color="lime")

                volume += y_mean**2 * np.pi * deltaL

                cylinders.append((x_0, deltaL, y_mean, "lime", 0.3))

                widths.append(y_mean)

            fig_volume = plt.figure(figsize=(6, 6))
            plt.scatter(
                (min(u1_adj) + min(u1_adj + deltaL)) / 2, y_mean, color="magenta", s=20
            )
            plt.plot([min(u1_adj), min(u1_adj)], [0, y_mean], color="lime")

            plt.axis("equal")
            plt.scatter(
                [i[0] for i in raw_points],
                [i[1] for i in raw_points],
                s=5,
                color="lime",
            )
            plt.xlabel("arc length")
            plt.ylabel("distance")
            plt.savefig(f"{self.dir_volume}/{self.cell_id}_volume.png", dpi=300)
            plt.savefig("realtime_volume.png")
            plt.close(fig_volume)
            Cell._plot_cylinders(
                cylinders, f"{self.dir_cylinders}/{self.cell_id}_cylinders.png"
            )
            # width はwidthsの大きい順から3つの平均値を取る。
            # widthsの各値は、その区間のy座標の平均値である。
            # この際、区間のy軸方向は細胞の片側の幅を表すため、値を単純に二倍する。
            widths = sorted(widths, reverse=True)
            widths = widths[:3]
            width = sum(widths) / len(widths)
            width *= 2
            return (area, volume, width, cell_length)

    def replot(self, calc_path: bool, degree: int, dir: str = "images") -> np.ndarray:
        mask = np.zeros_like(self.image_fluo_gray)

        cv2.fillPoly(mask, [pickle.loads(self.contour)], 255)

        coords_inside_cell_1 = np.column_stack(np.where(mask))
        points_inside_cell_1 = self.image_fluo_gray[
            coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
        ]

        X = np.array(
            [
                [i[1] for i in coords_inside_cell_1],
                [i[0] for i in coords_inside_cell_1],
            ]
        )

        if calc_path:
            (
                u1,
                u2,
                u1_contour,
                u2_contour,
                min_u1,
                max_u1,
                u1_c,
                u2_c,
                U,
                contour_U,
            ) = self._basis_conversion(
                [list(i[0]) for i in pickle.loads(self.contour)],
                X,
                self.image_fluo.shape[0] / 2,
                self.image_fluo.shape[1] / 2,
                coords_inside_cell_1,
            )

            fig = plt.figure(figsize=(6, 6))
            plt.scatter(u1, u2, s=5)
            plt.scatter(u1_c, u2_c, color="red", s=100)
            plt.axis("equal")
            margin_width = 50
            margin_height = 50
            plt.scatter(
                [i[1] for i in U],
                [i[0] for i in U],
                points_inside_cell_1,
                c=points_inside_cell_1,
                cmap="inferno",
                marker="o",
            )
            plt.xlim([min_u1 - margin_width, max_u1 + margin_width])
            plt.ylim([min(u2) - margin_height, max(u2) + margin_height])

            x = np.linspace(min_u1, max_u1, 1000)
            theta = self._poly_fit(U, degree=degree)
            y = np.polyval(theta, x)
            plt.plot(x, y, color="red")
            plt.scatter(u1_contour, u2_contour, color="lime", s=3)
            if dir is not None:
                if not os.path.exists(f"{dir}/replot"):
                    os.makedirs(f"{dir}/replot")
            plt.savefig(f"{dir}/replot/{self.cell_id}_replot.png")
            plt.close(fig)

            path_raw: list[Point] = self._find_path(
                self.cell_id, u1, u2, theta, points_inside_cell_1
            )
            path = [i.G for i in path_raw]
            return [(i - min(path)) / (max(path) - min(path)) for i in path]
        return []

    @staticmethod
    def _basis_conversion(
        contour: list[list[int]],
        X: np.ndarray,
        center_x: float,
        center_y: float,
        coordinates_incide_cell: list[list[int]],
    ) -> list[list[float]]:
        Sigma = np.cov(X)

        eigenvalues, eigenvectors = eig(Sigma)
        if eigenvalues[1] < eigenvalues[0]:
            Q = np.array([eigenvectors[1], eigenvectors[0]])
            U = [Q.transpose() @ np.array([i, j]) for i, j in coordinates_incide_cell]
            U = [[j, i] for i, j in U]
            contour_U = [Q.transpose() @ np.array([j, i]) for i, j in contour]
            contour_U = [[j, i] for i, j in contour_U]
            center = [center_x, center_y]
            u1_c, u2_c = center @ Q
        else:
            Q = np.array([eigenvectors[0], eigenvectors[1]])
            U = [
                Q.transpose() @ np.array([j, i]).transpose()
                for i, j in coordinates_incide_cell
            ]
            contour_U = [Q.transpose() @ np.array([i, j]) for i, j in contour]
            center = [center_x, center_y]
            u2_c, u1_c = center @ Q

        u1 = [i[1] for i in U]
        u2 = [i[0] for i in U]
        u1_contour = [i[1] for i in contour_U]
        u2_contour = [i[0] for i in contour_U]
        min_u1, max_u1 = min(u1), max(u1)

        return u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U

    @staticmethod
    def _poly_fit(U: list[list[float]], degree: int = 1) -> list[float]:
        u1_values = np.array([i[1] for i in U])
        f_values = np.array([i[0] for i in U])
        W = np.vander(u1_values, degree + 1)

        return inv(W.T @ W) @ W.T @ f_values

    @staticmethod
    def _calc_arc_length(theta: list[float], u_1_1: float, u_1_2: float) -> float:
        fx = lambda x: np.sqrt(
            1
            + sum(i * j * x ** (i - 1) for i, j in enumerate(theta[::-1][1:], start=1))
            ** 2
        )
        arc_length, _ = scipy.integrate.quad(fx, u_1_1, u_1_2, epsabs=1e-01)
        return arc_length

    @staticmethod
    def _find_minimum_distance_and_point(coefficients, x_Q, y_Q):
        # 関数の定義
        def f_x(x):
            return sum(
                coefficient * x**i for i, coefficient in enumerate(coefficients[::-1])
            )

        # 点Qから関数上の点までの距離 D の定義
        def distance(x):
            return np.sqrt((x - x_Q) ** 2 + (f_x(x) - y_Q) ** 2)

        # scipyのminimize関数を使用して最短距離を見つける
        # 初期値は0とし、精度は低く設定して計算速度を向上させる
        result = minimize(
            distance, 0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-2}
        )

        # 最短距離とその時の関数上の点
        x_min = result.x[0]
        min_distance = distance(x_min)
        min_point = (x_min, f_x(x_min))

        return min_distance, min_point

    @staticmethod
    def _find_path(
        cell_id: str,
        u1: list[float],
        u2: list[float],
        theta: list[float],
        points_inside_cell_1: list[float],
    ) -> list[Point]:

        ### projection
        raw_points: list[Point] = []
        for i, j, p in zip(u1, u2, points_inside_cell_1):
            min_distance, min_point = Cell._find_minimum_distance_and_point(theta, i, j)
            raw_points.append(Point(min_point[0], p))
        raw_points.sort()

        ### peak-path finder
        ### Meta parameters
        split_num: int = 35
        delta_L: float = (max(u1) - min(u1)) / split_num
        visualize: bool = True

        first_point: Point = raw_points[0]
        last_point: Point = raw_points[-1]
        path: list[Point] = [first_point]
        for i in range(1, int(split_num)):
            x_0 = min(u1) + i * delta_L
            x_1 = min(u1) + (i + 1) * delta_L
            points = [p for p in raw_points if x_0 <= p.u1 <= x_1]
            if len(points) == 0:
                continue
            point = max(points, key=lambda x: x.G)
            path.append(point)
        path.append(last_point)
        if visualize:
            fig = plt.figure(figsize=(6, 6))
            plt.axis("equal")
            plt.scatter(
                [i.u1 for i in raw_points],
                [i.G for i in raw_points],
                s=10,
                color="lime",
            )
            plt.scatter(
                [i.u1 for i in path],
                [i.G for i in path],
                s=50,
                color="magenta",
                zorder=100,
            )
            # print(len(path))
            plt.plot([i.u1 for i in path], [i.G for i in path], color="lime")
            fig.savefig(f"images/path/{cell_id}_path.png")
            fig.savefig("realtime_path.png")
            plt.close(fig)
        return path

    @staticmethod
    def _plot_cylinders(cylinders, out_path: str, resolution=50):
        """
        指定されたパラメータリストで複数のx軸に平行な円柱を描画する関数。

        Parameters:
            cylinders (list of tuples): 各円柱の設定リスト。
                                        各タプルは (start_x, height, radius, color, alpha) を含む。
            resolution (int): グリッドの解像度。

        使用例:
            cylinder_params = [
                (0, 5, 1, 'lime', 0.6),
                (6, 3, 0.5, 'lime', 0.8),
                (-4, 2, 2, 'lime', 0.5)
            ]
            plot_cylinders(cylinder_params)
        """
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        for start_x, height, radius, color, alpha in cylinders:
            # 円柱のパラメトリックな表現
            x = np.linspace(start_x, start_x + height, resolution)
            theta = np.linspace(0, 2 * np.pi, resolution)
            theta, x = np.meshgrid(theta, x)
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)

            # 各円柱をプロット
            ax.plot_surface(x, y, z, color=color, alpha=alpha, rcount=10, ccount=10)

            # 円柱の両端に円盤を描画
            theta = np.linspace(0, 2 * np.pi, resolution)
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)
            ax.plot(start_x * np.ones_like(theta), y, z, color=color, alpha=alpha)
            ax.plot(
                (start_x + height) * np.ones_like(theta), y, z, color=color, alpha=alpha
            )

            # 円柱の中心線を描画
            x = np.linspace(start_x, start_x + height, resolution)
            y = np.zeros_like(x)
            z = np.zeros_like(x)
            ax.plot(x, y, z, color="red")

        # 軸の設定
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax.set_zlabel("Z")

        # 目盛りのスケールを揃える
        max_range = max([abs(i[0]) for i in cylinders]) + max([i[1] for i in cylinders])
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        # グラフの余白を削除
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        fig.savefig(out_path, dpi=400)
        plt.close(fig)
