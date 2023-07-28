"""
- 관람객 데이터 .csv파일을 불러와서 heatmap으로 표현
- artwork 영역을 매트릭스로 저장한 후에 .npy 혹은 .npz 형태로 저장
"""
import copy
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Function to create an empty heatmap
def create_empty_heatmap(space_vertical_cells, space_horizontal_cells):
    return np.zeros((space_vertical_cells, space_horizontal_cells), dtype=np.uint16)


# Function to process visitor data and update the heatmap
def process_visitor_data(
    heatmap,
    dict_array,
    visitor_data_csv,
    x_offset,
    y_offset,
    heatmap_cell_size,
):
    reader = csv.DictReader(visitor_data_csv)

    for line in reader:
        new_x_cord = x_offset + float(line["move_x"])
        new_y_cord = y_offset + float(line["move_y"])

        for yx, i in np.ndenumerate(dict_array):
            if (
                yx[0] * heatmap_cell_size
                <= new_y_cord
                < (yx[0] + 1) * heatmap_cell_size
            ):
                if (
                    yx[1] * heatmap_cell_size
                    <= new_x_cord
                    < (yx[1] + 1) * heatmap_cell_size
                ):
                    if line["lookingAt"] != "wall":
                        if line["lookingAt"] not in dict_array[yx[0]][yx[1]]:
                            dict_array[yx[0]][yx[1]][line["lookingAt"]] = 1
                        else:
                            dict_array[yx[0]][yx[1]][line["lookingAt"]] += 1

        for yx, i in np.ndenumerate(heatmap):
            if (
                yx[0] * heatmap_cell_size
                <= new_y_cord
                < (yx[0] + 1) * heatmap_cell_size
            ):
                if (
                    yx[1] * heatmap_cell_size
                    <= new_x_cord
                    < (yx[1] + 1) * heatmap_cell_size
                ):
                    heatmap[yx[0]][yx[1]] += 1
                    print("processing")
                    print(yx)
    print("end")
    visitor_data_csv.close()


"""
예상 공간 값으로 0값 들어가는 이중배열 생성
"""
space_vertical_size, space_horizontal_size = 10, 15
# 공간 세로 길이: 30미터 | 공간 가로 길이: 60미터
heatmap_cell_size = 0.2
# 히트맵 셀 사이즈: 0.2미터 = 20센티미터
space_vertcal_cells, space_horizontal_cells = (
    round(space_vertical_size / heatmap_cell_size),
    round(space_horizontal_size / heatmap_cell_size),
)
# 히트맵 가로 셀 개수: 15/0.2 = 75개 | 히트맵 세로 셀 개수: 10/0.2 = 50개
heatmap = create_empty_heatmap(space_vertcal_cells, space_horizontal_cells)
# 히트맵 numpy 이중배열

array1 = [dict() for _ in range(space_vertcal_cells * space_horizontal_cells)]
dict_array = np.reshape(array1, (space_vertcal_cells, space_horizontal_cells))

"""
작품별 npy 저장하는 폴더 생성 및 작품리스트 생성
"""
current_path = os.getcwd()
visitor_data_csv = open("preAURA_1025_1030.csv", "r")
with open("Daegu_new.json", "r") as f:
    artwork_data_json = json.load(f)
    artwork_data_json_name = f.name[:-5]
# 관람객데이터 csv파일, 전시정보 json파일 open

artwork_list = [x["id"] for x in artwork_data_json["exhibitionObjects"]]
artwork_visitor_data_dir = artwork_data_json_name + "_" + visitor_data_csv.name[:-4]
os.makedirs(current_path + "/" + artwork_visitor_data_dir, exist_ok=True)
# "Daegu_new_preAURA_mmdd_MMDD" => AURA 이전의 mmdd 부터 MMDD 까지의 Daegu_new 콜렉션의 작품들 .npy 들을 보관하는 폴더명

reader = csv.DictReader(visitor_data_csv)

xOffset = -2.4
yOffset = 0
# 현재 5전시실만 배열로 변환
process_visitor_data(
    heatmap,
    dict_array,
    visitor_data_csv,
    xOffset,
    yOffset,
    heatmap_cell_size,
)

plt.figure(1)
sub_plots, axes = plt.subplots(1, 25, sharey=True)
num = 0

for artwork in artwork_list:
    artwork_np = create_empty_heatmap(space_vertcal_cells, space_horizontal_cells)
    for yx, i in np.ndenumerate(dict_array):  # 나중에 수정 필요... 임시방편
        if artwork in dict_array[yx[0]][yx[1]]:
            artwork_np[yx[0]][yx[1]] = int(dict_array[yx[0]][yx[1]][artwork])
    if np.max(artwork_np) != 0:
        np.save(
            current_path + "/" + artwork_visitor_data_dir + "/" + artwork, artwork_np
        )
        artwork_np_csv = pd.DataFrame(artwork_np)
        sns.heatmap(artwork_np_csv, cmap="Greens", vmin=0, vmax=200, ax=axes[num])
        num += 1

heatmap_csv = pd.DataFrame(heatmap)
dict_array_csv = pd.DataFrame(dict_array)
heatmap_csv.to_csv(visitor_data_csv.name[:-4] + "_Heatmap.csv", index=False)
dict_array_csv.to_csv(visitor_data_csv.name[:-4] + "_DictArray.csv", index=False)

# sns.heatmap(heatmapCSV, cmap='Greens', vmin=0, vmax=200)
# plt.figure(2)
# sns.heatmap(h25, cmap='BuPu')
# plt.figure(3)
# sns.heatmap(h24, cmap='PuRd')

plt.show()
