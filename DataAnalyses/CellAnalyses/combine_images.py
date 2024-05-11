import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def combine_images_function(
    image_size,
    out_name: str,
    input_dir: str,
):
    file_names = sorted(os.listdir(input_dir))
    num_images = len(file_names)
    total_rows = int(np.sqrt(num_images)) + 1
    total_cols = num_images // total_rows + 1
    result_image = np.zeros(
        (total_rows * image_size, total_cols * image_size, 3), dtype=np.uint8
    )
    # 画像をまとめる処理 Replot のフォルダから
    for i in range(total_rows):
        for j in range(total_cols):
            image_index = i * total_cols + j  # 画像のインデックス
            if image_index < num_images:
                image_path = (
                    f"{input_dir}/{file_names[image_index]}"  # 画像のパスを適切に設定
                )
                print(image_path)
                # 画像を読み込んでリサイズ
                img = cv2.imread(image_path)
                img = cv2.resize(img, (image_size, image_size))
                # まとめる画像に配置
                result_image[
                    i * image_size : (i + 1) * image_size,
                    j * image_size : (j + 1) * image_size,
                ] = img
    plt.axis("off")
    cv2.imwrite(f"{out_name}_combined.png", result_image)
    plt.close()
