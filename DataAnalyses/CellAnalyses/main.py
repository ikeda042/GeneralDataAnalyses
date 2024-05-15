from Cell import Cells
import numpy as np
from umap_analysis import cluster_analysis
from combine_images import combine_images_function
from migration import extract_1, extract_NA, extract_1_from_dbconsole
from tqdm import tqdm


def analyze_cells(
    db_path: str, morphology_analysis: bool, peak_path_analysis: bool, only_ph: bool
):
    cells: Cells = Cells(db_path=f"{db_path.split('.')[0]}.db", only_ph=only_ph)

    cell_ids = []
    areas = []
    volumes = []
    widths = []
    lengths = []
    paths = []

    for cell in tqdm(cells.get_cells()):
        cell.write_image(only_ph=only_ph)
        cell_ids.append(cell.cell_id)

        if morphology_analysis:
            area, volume, width, length = cell.replot_contour()
            areas.append(area)
            volumes.append(volume)
            widths.append(width)
            lengths.append(length)

        if peak_path_analysis:
            path: list[float] = cell.replot(calc_path=True, degree=4)
            paths.append(path)

    if morphology_analysis:
        with open(
            f"{db_path.split('.')[0]}_width_length_area_volume.csv", "w"
        ) as fpout:
            header = "width(px),length(px),area(px^2),volume(px^3)"
            fpout.write(header + "\n")
            for w, l, a, v in zip(widths, lengths, areas, volumes):
                fpout.write(f"{w},{l},{a},{v}\n")

    if peak_path_analysis:
        with open(f"{db_path.split('.')[0]}_paths.txt", "w") as fpout:
            for path, cell_id in zip(paths, cell_ids):
                if len(path) > 20:
                    fpout.write(f"{cell_id}|{','.join([str(i) for i in path])}\n")
                else:
                    print(f"cell_id: {cell_id} is too short path.")

    try:
        combine_images_function(200, f"{db_path.split('.')[0]}_images_ph", "images/ph")
    except:
        print("No images/ph directory")

    try:
        combine_images_function(
            200, f"{db_path.split('.')[0]}_images_replot", "images/replot"
        )
    except:
        print("No images/replot directory")

    try:
        combine_images_function(
            500, f"{db_path.split('.')[0]}_images_volume", "images/volume"
        )
    except:
        print("No images/volume directory")

    try:
        combine_images_function(
            500, f"{db_path.split('.')[0]}_images_cylinders", "images/cylinders"
        )
    except:
        print("No images/cylinders directory")

    try:
        combine_images_function(
            500, f"{db_path.split('.')[0]}_images_paths", "images/path"
        )
    except:
        print("No images/contour directory")


##########################################################################################################################################################################
# dbのパス(sqlite3、PhenoPixelから出力したデータベースもしくはCEll db consoleからダウンロードしたデータベース)
db_path = "DataAnalyses/CellAnalyses/demo-dataset/cell.db"

# 位相差モードのみの場合はTrue、蛍光二重レイヤを含む場合はFalse
only_ph = True

# 形態解析を行う場合はTrue、行わない場合はFalse
morphology_analysis = True

# peak-path解析を行う場合はTrue、行わない場合はFalse
peak_path_analysis = False


##########################################################################################################################################################################
# db　migration Migration.pyを参照
# PhenoPixelから出力したデータベースの場合は以下のマイグレーションを実行
# extract_NA(db_path)
# extract_1(db_path)

# CEll db consoleからダウンロードしたデータベースの場合は以下のマイグレーションを実行
# extract_1_from_dbconsole(db_path)
##########################################################################################################################################################################

# CellAnalysesを実行
if __name__ == "__main__":
    for d in [
        "DataAnalyses/CellAnalyses/dataset-space/SK25_LB_3mL 1_1.db",
        "DataAnalyses/CellAnalyses/dataset-space/SK25_LB_3mL 2_1.db",
        "DataAnalyses/CellAnalyses/dataset-space/SK25_LB_3mL 3_1.db",
    ]:
        analyze_cells(f"{d}", morphology_analysis, peak_path_analysis, only_ph)
