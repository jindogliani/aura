"""
- 관람객 데이터 .csv파일을 불러와서 heatmap으로 표현
- artwork 영역을 매트릭스로 저장한 후에 .npy 혹은 .npz 형태로 저장
"""

import copy
import csv
import json
import math
import os
from time import localtime, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize

date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) + ')'

# Function to create an empty heatmap
def create_empty_heatmap(space_vertical_cells, space_horizontal_cells):
    return np.zeros((space_vertical_cells, space_horizontal_cells), dtype=np.uint16)

# Function to process visitor data and update the heatmap
def process_heatmap(heatmap, dict_array, reader, x_offset, y_offset, unit_cell_size):
    for data in reader:
        x_cord = (x_offset + float(data["move_x"])) / unit_cell_size
        y_cord = (y_offset + float(data["move_y"])) / unit_cell_size #TODO #추후 수집하는 관람객 데이터 형태에 따라 수정 필요
        gaze_target = data["looking_at"]
        row, col = math.floor(y_cord), math.floor(x_cord)
        if row < 0 or col < 0 or row >= heatmap.shape[0] or col >= heatmap.shape[1]:
            continue
        dic = dict_array[row, col]
        if gaze_target != "wall":
            if gaze_target in dic:
                dict_array[row, col][gaze_target] += 1
            else:
                dict_array[row, col][gaze_target] = 1
            heatmap[row, col] += 1

def process_artwork_heatmap(
    artwork_id_list, rows, cols, dict_array, artwork_visitor_data_dir, axes, plot_column
):
    num = 0
    for artwork_id in artwork_id_list:
        artwork_heatmap = create_empty_heatmap(rows, cols)
        for (row, col), dic in np.ndenumerate(dict_array):
            if artwork_id in dic:
                artwork_heatmap[row, col] = int(dic[artwork_id])
        if np.max(artwork_heatmap) != 0:
            plt_row, plt_col = divmod(num, plot_column)
            save_file_path = os.path.join(artwork_visitor_data_dir, artwork_id)
            np.save(save_file_path, artwork_heatmap)
            artwork_heatmap_df = pd.DataFrame(artwork_heatmap)
            sns.heatmap(
                 artwork_heatmap_df, cmap="RdYlGn_r", vmin=0, vmax=30, ax=axes[plt_row, plt_col]
            )
            axes[plt_row, plt_col].set_title(artwork_id)
            print("processing", num, artwork_id)
            num += 1

"""
작품별 npy 저장하는 폴더 생성 및 작품리스트 생성
"""
cwd = os.getcwd()
artwork_data_path = "Daegu_new.json"
visitor_data_path = "VisitorData/preAURA_1025_1117.csv"
exhibition_data_path = "Data_2022.json"
artwork_data_filename, _ = os.path.splitext(os.path.basename(artwork_data_path))
visitor_data_filename, _ = os.path.splitext(os.path.basename(visitor_data_path))
artwork_visitor_data_dirname = f"{artwork_data_filename}_{visitor_data_filename}"
artwork_visitor_data_dir = os.path.join(cwd, artwork_visitor_data_dirname + date)
os.makedirs(artwork_visitor_data_dir, exist_ok=True)

"""
예상 공간 값으로 0값 들어가는 이중배열 생성
"""
#공간 세로 길이: 20미터 | 공간 가로 길이: 20미터
#히트맵 셀 사이즈: 0.1미터 = 10센티미터
#히트맵 가로 셀 개수: 20/0.1 = 200개 | 히트맵 세로 셀 개수: 20/0.1 = 200개 
spaceVerticalSize, spaceHorizontalSize = 20, 20
heatmapCellSize = 0.1
spaceVertcalCells, spaceHorizontalCells = spaceVerticalSize / heatmapCellSize, spaceHorizontalSize / heatmapCellSize
rows, cols = round(spaceVertcalCells), round(spaceHorizontalCells)

#현재는 공간데이터에 오프셋 값을 줌 # xOffset, yOffset = -4.1, -5.8
#2022년 공간데이터와 관람객 데이터 사이의 위치 차이
#관람객 시작점이 상대좌표에서 (0, y, 0) 이었으나 절대좌표에서는 (-4.1, y, -5.8)
with open(visitor_data_path, "r") as f:
    heatmap = create_empty_heatmap(rows, cols)
    dict_array = np.reshape([dict() for _ in range(rows * cols)], (rows, cols)) #TODO
    reader = csv.DictReader(f)
    x_offset, y_offset = 4-4.1, 10-5.8
    process_heatmap(heatmap, dict_array, reader, x_offset, y_offset, heatmapCellSize)

#내일 히트맵 찍어봐서 비교 필요함. 2023/08/06

plt_row, plt_col = 2, 7
_figs, axes = plt.subplots(plt_row, plt_col, sharey=False)

#전시정보 json파일 open
with open(artwork_data_path, "r") as f:
    artwork_data = json.load(f)
artwork_id_list = [artwork["id"] for artwork in artwork_data["exhibitionObjects"]] #TODOs

#2022년 기준 13작품만 관람객 데이터를 수집 함
hall5_exhibited_artwork_list = ["PA-0064", "PA-0067", "PA-0027", "PA-0025", "PA-0087", "PA-0070", "PA-0066", "PA-0045", "PA-0079", "PA-0024", "PA-0085", "PA-0063", "PA-0083"]

process_artwork_heatmap(
    hall5_exhibited_artwork_list, rows, cols, dict_array, artwork_visitor_data_dir, axes, plt_col
)
plt.show()

np.save(artwork_visitor_data_dir + "/" + visitor_data_filename + "_Heatmap" + date, heatmap)
heatmap_csv = pd.DataFrame(heatmap)
dict_array_csv = pd.DataFrame(dict_array)
heatmap_csv.to_csv(artwork_visitor_data_dir + "/" + visitor_data_filename + "_Heatmap" + date + ".csv", index=False)
dict_array_csv.to_csv(artwork_visitor_data_dir + "/" + visitor_data_filename + "_DictArray" + date + ".csv", index=False)

sns.heatmap(heatmap_csv, cmap='RdYlGn_r', vmin=0, vmax=150)
plt.show()