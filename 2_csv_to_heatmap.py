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

date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) +'-'+ str(localtime(time()).tm_hour) + '-'+ str(localtime(time()).tm_min) + ')'
user_list = []

# Function to create an empty heatmap
def create_empty_heatmap(space_vertical_cells, space_horizontal_cells):
    return np.zeros((space_vertical_cells, space_horizontal_cells), dtype=np.uint16)

# Function to process visitor data and update the heatmap
def process_heatmap(heatmap, dict_array, reader, x_offset, y_offset, unit_cell_size):
    for data in reader:
        if data["id"] not in user_list:
            user_list.append(data["id"])
        x_cord = (x_offset + float(data["position_x"])) / unit_cell_size
        y_cord = (y_offset + float(data["position_z"])) / unit_cell_size #TODO #추후 수집하는 관람객 데이터 형태에 따라 수정 필요
        gaze_target = data["looking_at"] #TODO
        row, col = math.floor(y_cord), math.floor(x_cord)
        if row < 0 or col < 0 or row >= heatmap.shape[0] or col >= heatmap.shape[1]:
            continue
        dic = dict_array[row, col]
        if gaze_target != "none": #TODO
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
                 artwork_heatmap_df, cmap="RdYlGn_r", vmin=0, vmax=100, ax=axes[plt_row, plt_col]
            )
            axes[plt_row, plt_col].set_title(artwork_id)
            print("processing", num, artwork_id)
            num += 1

def visitor_csv_to_heatmap(ver, exhibition_data_path, visitor_data_path, heatmap_cell_size, space_vertical_size, space_horizontal_size, x_offset, z_offset):
    
    exhibition_data_filename, _ = os.path.splitext(os.path.basename(exhibition_data_path))
    visitor_data_filename, _ = os.path.splitext(os.path.basename(visitor_data_path))
    exhibition_visitor_data_dirname = f"{exhibition_data_filename}_{visitor_data_filename}"
    exhibition_visitor_data_dir = os.path.join(cwd, exhibition_visitor_data_dirname + date)
    os.makedirs(exhibition_visitor_data_dir, exist_ok=True)

    space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
    rows, cols = round(space_vertcal_cells), round(space_horizontal_cells)

    with open(visitor_data_path, "r", encoding='utf-8-sig') as f:
        heatmap = create_empty_heatmap(rows, cols)
        dict_array = np.reshape([dict() for _ in range(rows * cols)], (rows, cols))
        reader = csv.DictReader(f)
        process_heatmap(heatmap, dict_array, reader, x_offset, z_offset, heatmap_cell_size)

    plt_row, plt_col = 5, 9
    _figs, axes = plt.subplots(plt_row, plt_col, sharey=False)

    with open(exhibition_data_path, "r", encoding='utf-8-sig') as f:
        exhibition_data = json.load(f)
    artwork_id_list = [artwork["artworkIndex"] for artwork in exhibition_data["paintings"]] #2023

    process_artwork_heatmap( #TODO 2023년에는 artwork_id_list 를 매개변수로 해야 함
        artwork_id_list, rows, cols, dict_array, exhibition_visitor_data_dir, axes, plt_col
    )
    plt.show()

    np.save(exhibition_visitor_data_dir + "/" + visitor_data_filename + "_Heatmap" + date, heatmap)

    heatmap_csv = pd.DataFrame(heatmap)
    dict_array_csv = pd.DataFrame(dict_array)
    heatmap_csv.to_csv(exhibition_visitor_data_dir + "/" + visitor_data_filename + "_Heatmap" + date + ".csv", index=False)
    dict_array_csv.to_csv(exhibition_visitor_data_dir + "/" + visitor_data_filename + "_DictArray" + date + ".csv", index=False)

    print("user number : " + str(len(user_list)))
    sns.heatmap(heatmap_csv, cmap='RdYlGn_r', vmin=0, vmax=50)
    plt.show()

ver = "2022"
cwd = os.getcwd()
artwork_data_path = "Daegu_new.json"

if ver == "2023":
    exhibition_data_path = "Data_2023.json"
    visitor_data_path = "VisitorData2023/preAURA_2023.csv"
    
    space_vertical_size, space_horizontal_size = 40, 40
    heatmap_cell_size = 0.1
    x_offset, z_offset = 7, 12 
    visitor_csv_to_heatmap(ver, exhibition_data_path, visitor_data_path, heatmap_cell_size, space_vertical_size, space_horizontal_size, x_offset, z_offset)

elif ver == "2022":
    exhibition_data_path = "Data_2022.json"
    visitor_data_path = "VisitorData2022/preAURA_2022.csv"

    space_vertical_size, space_horizontal_size = 40, 40
    heatmap_cell_size = 0.1
    x_offset, z_offset = 25, 20
    visitor_csv_to_heatmap(ver, exhibition_data_path, visitor_data_path, heatmap_cell_size, space_vertical_size, space_horizontal_size, x_offset, z_offset) 