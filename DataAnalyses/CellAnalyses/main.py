from Cell import Cells
import numpy as np
from umap_analysis import cluster_analysis
from combine_images import combine_images_function
from migration import extract_1, extract_NA, extract_1_from_dbconsole
from tqdm import tqdm

def analyze_cells(db_path: str, morphology_analysis: bool, peak_path_analysis: bool, only_ph: bool):
    cells: Cells = Cells(db_path=f"{db_path.split('.')[0]}.db", only_ph=only_ph)

    cell_ids = []
    areas = []
    volumes = []
    widths = []
    paths = []

    for cell in tqdm(cells.get_cells()):
        cell.write_image(only_ph=only_ph)
        cell_ids.append(cell.cell_id)

        if morphology_analysis:
            area, volume, width = cell.replot_contour()
            areas.append(area)
            volumes.append(volume)
            widths.append(width)
        
        if peak_path_analysis:
            path: list[float] = cell.replot(calc_path=True, degree=4)
            paths.append(path)

    if morphology_analysis:
        with open(f"{db_path.split('.')[0]}_width_area_volume.csv", "w") as fpout:
            header = "width(px),area(px^2),volume(px^3)"
            fpout.write(header + "\n")
            for w, a, v in zip(widths, areas, volumes):
                fpout.write(f"{w},{a},{v}\n")

    if peak_path_analysis:
        with open(f"{db_path.split('.')[0]}_paths.txt", "w") as fpout:
            for path,cell_id in zip(paths, cell_ids):
                if len(path) > 20:
                    fpout.write(f"{cell_id}|{','.join([str(i) for i in path])}\n")
                else:
                    print(f"cell_id: {cell_id} is too short path.")

    try:
        combine_images_function(200, f"{db_path.split('.')[0]}_images_ph", "images/ph")
    except:
        print("No images/ph directory")

    try:
        combine_images_function(200, f"{db_path.split('.')[0]}_images_replot", "images/replot")
    except:
        print("No images/replot directory")

    try:
        combine_images_function(500, f"{db_path.split('.')[0]}_images_volume", "images/volume")
    except:
        print("No images/volume directory")

    try:
        combine_images_function(500, f"{db_path.split('.')[0]}_images_cylinders", "images/cylinders")
    except:
        print("No images/cylinders directory")

    try:
        combine_images_function(500, f"{db_path.split('.')[0]}_images_paths", "images/path")
    except:
        print("No images/contour directory")



##########################################################################################################################################################################
# dbのパス(sqlite3、PhenoPixelから出力したデータベースもしくはCEll db consoleからダウンロードしたデータベース)
db_path = "dataset-space/sk320cip/sk320cip0min.db"

# 位相差モードのみの場合はTrue、蛍光二重レイヤを含む場合はFalse
only_ph = False

# 形態解析を行う場合はTrue、行わない場合はFalse
morphology_analysis = False

# peak-path解析を行う場合はTrue、行わない場合はFalse
peak_path_analysis = True


##########################################################################################################################################################################
# db　migration Migration.pyを参照
# PhenoPixelから出力したデータベースの場合は以下のマイグレーションを実行
# extract_NA(db_path)
# extract_1(db_path)

#CEll db consoleからダウンロードしたデータベースの場合は以下のマイグレーションを実行
# extract_1_from_dbconsole(db_path)
##########################################################################################################################################################################

# CellAnalysesを実行
if __name__ == "__main__":
    ab_tag = "gen"
    db_paths = [f"dataset-space/sk320{ab_tag}/sk320{ab_tag}0min.db", 
                f"dataset-space/sk320{ab_tag}/sk320{ab_tag}30min.db", 
                f"dataset-space/sk320{ab_tag}/sk320{ab_tag}60min.db",
                f"dataset-space/sk320{ab_tag}/sk320{ab_tag}90min.db",
                f"dataset-space/sk320{ab_tag}/sk320{ab_tag}120min.db",
                ]
    for db_path in db_paths: 
        extract_1_from_dbconsole(db_path)
        analyze_cells(db_path, morphology_analysis, peak_path_analysis, only_ph)
