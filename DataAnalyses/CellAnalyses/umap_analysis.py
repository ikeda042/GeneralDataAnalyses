import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os 
import shutil
from combine_images import combine_images_function
from Cell import Cells

class PeakPath:
    def __init__(self, cell_id: str, path: list[float]):
        self.cell_id = cell_id
        self.path = path

def localization_clustering(filename: str):
    with open(filename, "r") as fp:
        lines = fp.readlines()
        paths = []
        for line in lines:
            cell_id, path = line.strip().split("|")
            path = [float(i) for i in path.split(",")]
            if len(path) > 20: 
                paths.append(PeakPath(cell_id, path))

    # UMAPを使用してデータを低次元に変換
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform([i.path for i in paths])

    # KMeans クラスタリング
    kmeans = KMeans(n_clusters=3, random_state=42).fit(embedding)

    # クラスタリングの結果を可視化
    plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans.labels_, s=30)
    plt.title('2D Embedding by UMAP')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.savefig("umap.png", dpi=500)

    clusters = {label: [] for label in  set(kmeans.labels_)}

    # 各cell_idとそのクラスター番号を出力
    for path, label in zip(paths, kmeans.labels_):
        clusters[label].append(path.cell_id)

    return clusters


def cluster_analysis(paths_file:str, cells: Cells) -> None:
    clusters = localization_clustering(paths_file)
    for cluster in clusters:
        if not os.path.exists(f"cluster_{cluster}"):
            os.mkdir(f"cluster_{cluster}")
        else:
            for file in os.listdir(f"cluster_{cluster}"):
                shutil.rmtree(f"cluster_{cluster}/{file}")  
        cell_ids = clusters[cluster]
        for cell_id in cell_ids:
            cell = cells.get_cell(cell_id)
            cell.write_image(f"cluster_{cluster}")
            cell.replot(dir = f"cluster_{cluster}",calc_path = False)
        combine_images_function(200, f"cluster_{cluster}", f"cluster_{cluster}/replot")
        