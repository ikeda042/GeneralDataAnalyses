from Cell import Cells
import numpy as np
from umap_analysis import cluster_analysis
from combine_images import combine_images_function
from migration import extract_1, extract_NA, extract_1_from_dbconsole

##########################################################################################################################################################################
# dbのパス(sqlite3、PhenoPixelから出力したデータベースもしくはCEll db consoleからダウンロードしたデータベース)
db_path = "sk320cip/sk320cip0min.db"

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
extract_1_from_dbconsole(db_path)
##########################################################################################################################################################################

cells: Cells = Cells(db_path=f"{db_path.split(".")[0]}.db",only_ph=only_ph)

# 細胞IDの取得
cell_ids = []
# 形態パラメータ保持
areas = []
volumes = []
widths = []
# peak-path 解析用のpath保持
paths = []

for cell in cells.get_cells():
    # PHのみのモードと蛍光二重レイヤを選択するs
    cell.write_image(only_ph=only_ph)
    cell_ids.append(cell.cell_id)

    # 細胞輪郭を用いて面積、体積、幅を計算
    if morphology_analysis:
        area, volume, width = cell.replot_contour()
        areas.append(area)
        volumes.append(volume)
        widths.append(width)
    
    # peak-path解析
    if peak_path_analysis:
        path: list[float] = cell.replot(calc_path=True, degree=4)
        paths.append(path)

if morphology_analysis:
    # 形態解析の結果をcsvファイルに保存
    with open(f"{db_path.split(".")[0]}_width_area_volume.csv", "w") as fpout:
        header = "width(px),area(px^2),volume(px^3)"
        fpout.write(header + "\n")
        for w, a, v in zip(widths, areas, volumes):
            fpout.write(f"{w},{a},{v}\n")

# UMAPによるクラスター解析(pathの計算が終了していることを前提とする)
# pathは一旦txtファイルに保存しておく。この時、cell_id|pathの形式で保存する
if peak_path_analysis:
    with open(f"{db_path.split(".")[0]}_paths.txt", "w") as fpout:
        for path,cell_id in zip(paths, cell_ids):
            if len(path) > 20:
                fpout.write(f"{cell_id}|{'.'.join(path)}\n")
            else:
                print(f"cell_id: {cell_id} is too short path.")
    cluster_analysis(f"{db_path.split(".")[0]}_paths.txt", cells)

# 画像の結合
try:
    combine_images_function(200, f"{db_path.split(".")[0]}_images_ph", "images/ph")
except:
    print("No images/ph directory")

try:
    combine_images_function(200, f"{db_path.split(".")[0]}_images_replot", "images/replot")
except:
    print("No images/replot directory")

try:
    combine_images_function(500, f"{db_path.split(".")[0]}_images_volume", "images/volume")
except:
    print("No images/volume directory")
try:
    combine_images_function(500, f"{db_path.split(".")[0]}_images_cylinders", "images/cylinders")
except:
    print("No images/cylinders directory")



