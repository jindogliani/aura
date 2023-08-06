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
예상 공간 값으로 0값 들어가는 이중배열 생성
"""
#공간 세로 길이: 20미터 | 공간 가로 길이: 20미터
#히트맵 셀 사이즈: 0.1미터 = 10센티미터
#히트맵 가로 셀 개수: 20/0.1 = 200개 | 히트맵 세로 셀 개수: 20/0.1 = 200개 
spaceVerticalSize, spaceHorizontalSize = 10, 20
heatmapCellSize = 0.1
spaceVertcalCells, spaceHorizontalCells = spaceVerticalSize / heatmapCellSize, spaceHorizontalSize / heatmapCellSize
spaceVertcalCells, spaceHorizontalCells = round(spaceVertcalCells), round(spaceHorizontalCells)

heatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint16)


dic = dict()
array1 = []
for i in range(spaceVertcalCells*spaceHorizontalCells):
        array1.append(copy.deepcopy(dic))
        i += 1
dictArray = np.reshape(array1, (spaceVertcalCells, spaceHorizontalCells))
#나중에 수정 필요... 임시방편

"""
작품별 npy 저장하는 폴더 생성 및 작품리스트 생성
"""
cwd = os.getcwd()
artwork_data_path = "Daegu_new.json"
visitor_data_path = "VisitorData/preAURA_1025_1030.csv"
artwork_data_filename, _ = os.path.splitext(os.path.basename(artwork_data_path))
visitor_data_filename, _ = os.path.splitext(os.path.basename(visitor_data_path))
artwork_visitor_data_dirname = f"{artwork_data_filename}_{visitor_data_filename}"
artwork_visitor_data_dir = os.path.join(cwd, artwork_visitor_data_dirname)
os.makedirs(artwork_visitor_data_dir, exist_ok=True)




visitorDataCSV = open('VisitorData/preAURA_1025_1030.csv', 'r')
visitorDataCSVname = visitorDataCSV.name[12:-4]
reader = csv.DictReader(visitorDataCSV)

with open('Daegu_new.json', 'r') as f:
        artworkDataJSON = json.load(f)
        artworkDataJSONname = f.name[:-5]
#관람객데이터 csv파일, 전시정보 json파일 open

ArtworkList = []
for x in artworkDataJSON['exhibitionObjects']:
        ArtworkList.append(x['id'])
#ArtworkList 리스트에 작품들 관리번호 호출

artworkVisitorDataDir = artworkDataJSONname + '_' + visitorDataCSVname
os.makedirs(cwd+ "/" + artworkVisitorDataDir, exist_ok= True)
#"Daegu_new_preAURA_mmdd_MMDD" => AURA 이전의 mmdd 부터 MMDD 까지의 Daegu_new 콜렉션의 작품들 .npy 들을 보관하는 폴더명 

xOffset, yOffset = 0, 0 #현재는 공간데이터에 오프셋 값을 줌
# xOffset, yOffset = -4.1, -5.8
#2022년 공간데이터와 관람객 데이터 사이의 위치 차이
#관람객 시작점이 상대좌표에서 (0, y, 0) 이었으나 절대좌표에서는 (-4.1, y, -5.8)

for line in reader: 
      newXcord = xOffset + float(line['move_x'])
      newYcord = yOffset - float(line['move_y'])
      # for (y, x), i in np.ndenumerate(dictArray): 
      for yx, i in np.ndenumerate(dictArray): #나중에 수정 필요... 임시방편 #TODO
        if(yx[0]*heatmapCellSize <= newYcord < (yx[0]+1)*heatmapCellSize):
               if(yx[1]*heatmapCellSize <= newXcord < (yx[1]+1)*heatmapCellSize):
                     if line['lookingAt'] != 'wall':
                        if line['lookingAt'] not in dictArray[yx[0]][yx[1]]:
                              dictArray[yx[0]][yx[1]][line['lookingAt']] = 1
                        elif line['lookingAt'] in dictArray[yx[0]][yx[1]]:
                              dictArray[yx[0]][yx[1]][line['lookingAt']] += 1                      
      for yx, i in np.ndenumerate(heatmap): #나중에 수정 필요... 임시방편
            if(yx[0]*heatmapCellSize <= newYcord < (yx[0]+1)*heatmapCellSize):
                   if(yx[1]*heatmapCellSize <= newXcord < (yx[1]+1)*heatmapCellSize):
                          heatmap[yx[0]][yx[1]] += 1
                          print(yx)      
print('end')
visitorDataCSV.close()


plt.figure(1)
# sub_plots, axes = plt.subplots(1, 25, sharey=True)
# num = 0
for artwork in ArtworkList:
      artworkNp = artwork + 'Heatmap'
      globals()[artworkNp] = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint16)
      artwork_Np = globals()[artworkNp]
      for yx, i in np.ndenumerate(dictArray): #나중에 수정 필요... 임시방편
            if artwork in dictArray[yx[0]][yx[1]]:
                  artwork_Np[yx[0]][yx[1]] = int(dictArray[yx[0]][yx[1]][artwork])
      if np.max(artwork_Np) !=0:
            np.save(cwd + "/" +artworkVisitorDataDir + "/" + artwork, artwork_Np)
            artwork_Np_CSV = pd.DataFrame(artwork_Np)
            # sns.heatmap(artwork_Np_CSV, cmap='Greens', vmin=0, vmax=200, ax=axes[num])
            # num += 1

np.save(cwd + "/" +artworkVisitorDataDir + "/" + visitorDataCSVname + date, heatmap)
heatmapCSV = pd.DataFrame(heatmap)
dictArrayCSV = pd.DataFrame(dictArray)
heatmapCSV.to_csv(visitorDataCSV.name[:-4] + '_Heatmap.csv', index=False)
dictArrayCSV.to_csv(visitorDataCSV.name[:-4] + '_DictArray.csv', index=False)

sns.heatmap(heatmapCSV, cmap='Greens', vmin=0, vmax=200)
plt.show()