"""
- 관람객 데이터 .csv파일을 불러와서 heatmap으로 표현
- artwork 영역을 매트릭스로 저장한 후에 .npy 혹은 .npz 형태로 저장
"""
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Function to create an empty heatmap
def create_empty_heatmap(space_vertical_cells, space_horizontal_cells):
    return np.zeros((space_vertical_cells, space_horizontal_cells), dtype=np.uint16)


# Function to process visitor data and update the heatmap
def process_heatmap(heatmap, dict_array, reader, x_offset, y_offset, unit_cell_size):
    for data in reader:
        x_cord = (float(data["move_x"]) + x_offset) / unit_cell_size
        y_cord = (float(data["move_y"]) + y_offset) / unit_cell_size
        gaze_target = data["lookingAt"]
        row, col = math.floor(y_cord), math.floor(x_cord)
        if row < 0 or col < 0 or row >= heatmap.shape[0] or col >= heatmap.shape[1]:
            continue
        dic = dict_array[row, col]

        # for (row, col), dic in np.ndenumerate(dict_array):
        #     if row <= y_cord < row + 1 and col <= x_cord < col + 1:
        if gaze_target != "wall":
            if gaze_target in dic:
                dict_array[row, col][gaze_target] += 1
            else:
                dict_array[row, col][gaze_target] = 1
                # else:
                #     continue

        # for (row, col), _ in np.ndenumerate(heatmap):
        # if row <= y_cord < row + 1 and col <= x_cord < col + 1:
        heatmap[row, col] += 1
        print("processing")
        print((row, col))
        # else:
        #     continue
    print("end")


def process_artwork_heatmap(
    artwork_id_list, rows, cols, dict_array, artwork_visitor_data_dir, axes
):
    num = 0
    for artwork_id in artwork_id_list:
        artwork_heatmap = create_empty_heatmap(rows, cols)
        for (row, col), dic in np.ndenumerate(dict_array):
            if artwork_id in dic:
                artwork_heatmap[row, col] = int(dic[artwork_id])
        if np.max(artwork_heatmap) != 0:
            save_file_path = os.path.join(artwork_visitor_data_dir, artwork_id)
            np.save(save_file_path, artwork_heatmap)
            artwork_heatmap_df = pd.DataFrame(artwork_heatmap)
            sns.heatmap(
                artwork_heatmap_df, cmap="Greens", vmin=0, vmax=200, ax=axes[num]
            )
            print("processing", num, artwork_id)
            num += 1


"""
작품별 npy 저장하는 폴더 생성 및 작품리스트 생성
"""
# "Daegu_new_preAURA_mmdd_MMDD" => AURA 이전의 mmdd 부터 MMDD 까지의 Daegu_new 콜렉션의 작품들 .npy 들을 보관하는 폴더명
cwd = os.getcwd()
artwork_data_path = "Daegu_new.json"
visitor_data_path = "preAURA_1025_1030.csv"
artwork_data_filename, _ = os.path.splitext(os.path.basename(artwork_data_path))
visitor_data_filename, _ = os.path.splitext(os.path.basename(visitor_data_path))
artwork_visitor_data_dirname = f"{artwork_data_filename}_{visitor_data_filename}"
artwork_visitor_data_dir = os.path.join(cwd, artwork_visitor_data_dirname)
os.makedirs(artwork_visitor_data_dir, exist_ok=True)

"""
예상 공간 값으로 0값 들어가는 이중배열 생성
"""
# 공간 세로 길이: 30미터 | 공간 가로 길이: 60미터
# 히트맵 셀 사이즈: 0.2미터 = 20센티미터
# 히트맵 가로 셀 개수: 15/0.2 = 75개 | 히트맵 세로 셀 개수: 10/0.2 = 50개
height, width = 10, 15  # horizontal, vertical length in meters
unit_cell_size = 0.2  # length of each cell in meters
rows, cols = round(height / unit_cell_size), round(width / unit_cell_size)

with open(visitor_data_path, "r") as f:
    heatmap = create_empty_heatmap(rows, cols)
    dict_array = np.reshape([dict() for _ in range(rows * cols)], (rows, cols))
    reader = csv.DictReader(f)
    x_offset = -2.4
    y_offset = 0
    process_heatmap(heatmap, dict_array, reader, x_offset, y_offset, unit_cell_size)

plt.figure(1)
_figs, axes = plt.subplots(1, 25, sharey=True)

with open(artwork_data_path, "r") as f:
    artwork_data = json.load(f)
artwork_id_list = [artwork["id"] for artwork in artwork_data["exhibitionObjects"]]

process_artwork_heatmap(
    artwork_id_list, rows, cols, dict_array, artwork_visitor_data_dir, axes
)

heatmap_csv = pd.DataFrame(heatmap)
dict_array_csv = pd.DataFrame(dict_array)
heatmap_csv.to_csv(visitor_data_filename + "_Heatmap.csv", index=False)
dict_array_csv.to_csv(visitor_data_filename + "_DictArray.csv", index=False)

plt.show()
