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
import bisect
from dataclasses import dataclass

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
    def __init__(self, db_path: str = "database_sk326.db") -> None:
        Base = declarative_base()

        class Cell_(Base):
            __tablename__ = "cells"
            id = Column(Integer, primary_key=True)
            cell_id = Column(String)
            img_ph = Column(BLOB)
            img_fluo1 = Column(BLOB)
            contour = Column(BLOB)

        engine: create_engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        Session: sessionmaker = sessionmaker(bind=engine)
        session = Session()
        cells: list[Cell_] = session.query(Cell_).all()
        print("Loading cells from database...")
        self.cells: list[Cell] = [
            Cell(
                cell.cell_id,
                image_ph=cv2.imdecode(
                    np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_COLOR
                ),
                image_fluo=cv2.imdecode(
                    np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_COLOR
                ),
                contour=cell.contour,
            )
            for cell in cells
        ]

    def get_cell(self, cell_id: str) -> Cell | None:
        cell_ids: list[str] = [cell.cell_id for cell in self.cells]
        index: int = bisect.bisect_left(cell_ids, cell_id)
        if index != len(self.cells) and self.cells[index].cell_id == cell_id:
            return self.cells[index]
        else:
            print(f"Cell with cell id {cell_id} not found")
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
    ) -> None:
        self.cell_id: str = cell_id
        self.image_ph: np.ndarray = image_ph
        self.image_fluo: np.ndarray = image_fluo
        self.contour: np.ndarray = contour
        self.image_fluo_gray: np.ndarray = cv2.cvtColor(
            self.image_fluo, cv2.COLOR_BGR2GRAY
        )

        # make directories
        self.dir_base: str = "images"
        self.dir_ph: str = f"{self.dir_base}/ph"
        self.dir_fluo: str = f"{self.dir_base}/fluo"
        self.dir_replot: str = f"{self.dir_base}/replot"
        self.dir_path: str = f"{self.dir_base}/path"
        if not os.path.exists(self.dir_base):
            os.makedirs(self.dir_base)
        if not os.path.exists(self.dir_ph):
            os.makedirs(self.dir_ph)
        if not os.path.exists(self.dir_fluo):
            os.makedirs(self.dir_fluo)
        if not os.path.exists(self.dir_replot):
            os.makedirs(self.dir_replot)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def __repr__(self) -> str:
        return f"Cell ID: {self.cell_id}"

    def write_image(self) -> None:
        out_name_ph: str = f"{self.dir_ph}/{self.cell_id}_ph.png"
        out_name_fluo: str = f"{self.dir_fluo}/{self.cell_id}_fluo.png"
        image_ph_copy: np.ndarray = self.image_ph.copy()
        image_fluo_copy: np.ndarray = self.image_fluo.copy()
        cv2.drawContours(image_ph_copy, pickle.loads(self.contour), -1, (0, 255, 0), 1)
        cv2.drawContours(
            image_fluo_copy, pickle.loads(self.contour), -1, (0, 255, 0), 1
        )
        cv2.imwrite(out_name_ph, image_ph_copy)
        cv2.imwrite(out_name_fluo, image_fluo_copy)
        print(f"Images written to {out_name_ph} and {out_name_fluo}")

    def get_contour(self) -> np.ndarray:
        return self.contour

    def get_image_ph(self) -> np.ndarray:
        return self.image_ph

    def get_image_fluo(self) -> np.ndarray:
        return self.image_fluo

    def replot(self) -> np.ndarray:
        mask = np.zeros_like(self.image_fluo_gray)

        cv2.fillPoly(mask, [pickle.loads(self.contour)], 255)

        coords_inside_cell_1 = np.column_stack(np.where(mask))
        points_inside_cell_1 = []
        for i in coords_inside_cell_1:
            points_inside_cell_1.append(self.image_fluo_gray[i[0], i[1]])

        X = np.array(
            [
                [i[1] for i in coords_inside_cell_1],
                [i[0] for i in coords_inside_cell_1],
            ]
        )
        print(X)
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
        theta = self._poly_fit(U)
        y = np.polyval(theta, x)
        plt.plot(x, y, color="red")
        plt.scatter(u1_contour, u2_contour, color="lime", s=3)
        plt.savefig(f"{self.dir_replot}/{self.cell_id}_replot.png")
        plt.savefig(f"realtime_replot.png")
        plt.close(fig)

        return self._find_path(self.cell_id, u1, u2, theta, points_inside_cell_1)

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
    def _poly_fit(U: list[list[float]], degree: int = 2) -> list[float]:
        u1_values = np.array([i[1] for i in U])
        f_values = np.array([i[0] for i in U])
        W = np.vander(u1_values, degree + 1)

        theta = inv(W.T@ W) @ W.T @ f_values
        print(theta)
        return theta

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
        split_num : int = 35
        delta_L : float = (max(u1) - min(u1)) / split_num
        visualize : bool= True

        first_point : Point = raw_points[0]
        last_point : Point = raw_points[-1]
        path : list[Point] = [first_point]
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
                color="lime"
            )
            plt.scatter(
                [i.u1 for i in path],
                [i.G for i in path],
                s=50,
                color="magenta",
                zorder=100,
            )
            print(len(path))
            plt.plot([i.u1 for i in path], [i.G for i in path], color="lime")
            fig.savefig(f"images/path/{cell_id}_path.png")
            fig.savefig("realtime_path.png")
            plt.close(fig)
        return path



if __name__ == "__main__":
    cells: Cells = Cells(db_path="demodata/cell.db")

    for cell in cells.get_cells():
        cell.write_image()
        cell.replot()
