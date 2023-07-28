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
def create_empty_heatmap(spaceVerticalCells, spaceHorizontalCells):
    return np.zeros((spaceVerticalCells, spaceHorizontalCells), dtype=np.uint16)


# Function to process visitor data and update the heatmap
def process_visitor_data(
    heatmap, dictArray, visitorDataCSV, xOffset, yOffset, heatmapCellSize, ArtworkList
):
    reader = csv.DictReader(visitorDataCSV)

    for line in reader:
        newXcord = xOffset + float(line["move_x"])
        newYcord = yOffset + float(line["move_y"])

        for yx, i in np.ndenumerate(dictArray):
            if yx[0] * heatmapCellSize <= newYcord < (yx[0] + 1) * heatmapCellSize:
                if yx[1] * heatmapCellSize <= newXcord < (yx[1] + 1) * heatmapCellSize:
                    if line["lookingAt"] != "wall":
                        if line["lookingAt"] not in dictArray[yx[0]][yx[1]]:
                            dictArray[yx[0]][yx[1]][line["lookingAt"]] = 1
                        else:
                            dictArray[yx[0]][yx[1]][line["lookingAt"]] += 1

        for yx, i in np.ndenumerate(heatmap):
            if yx[0] * heatmapCellSize <= newYcord < (yx[0] + 1) * heatmapCellSize:
                if yx[1] * heatmapCellSize <= newXcord < (yx[1] + 1) * heatmapCellSize:
                    heatmap[yx[0]][yx[1]] += 1
                    print("processing")
                    print(yx)
    print("end")
    visitorDataCSV.close()


"""
예상 공간 값으로 0값 들어가는 이중배열 생성
"""
spaceVerticalSize, spaceHorizontalSize = 10, 15
# 공간 세로 길이: 30미터 | 공간 가로 길이: 60미터
heatmapCellSize = 0.2
# 히트맵 셀 사이즈: 0.2미터 = 20센티미터
spaceVertcalCells, spaceHorizontalCells = (
    round(spaceVerticalSize / heatmapCellSize),
    round(spaceHorizontalSize / heatmapCellSize),
)
# 히트맵 가로 셀 개수: 15/0.2 = 75개 | 히트맵 세로 셀 개수: 10/0.2 = 50개
heatmap = create_empty_heatmap(spaceVertcalCells, spaceHorizontalCells)
# 히트맵 numpy 이중배열

array1 = [dict() for _ in range(spaceVertcalCells * spaceHorizontalCells)]
dictArray = np.reshape(array1, (spaceVertcalCells, spaceHorizontalCells))

"""
작품별 npy 저장하는 폴더 생성 및 작품리스트 생성
"""
currentPath = os.getcwd()
visitorDataCSV = open("preAURA_1025_1030.csv", "r")
with open("Daegu_new.json", "r") as f:
    artworkDataJSON = json.load(f)
    artworkDataJSONname = f.name[:-5]
# 관람객데이터 csv파일, 전시정보 json파일 open

ArtworkList = [x["id"] for x in artworkDataJSON["exhibitionObjects"]]
artworkVisitorDataDir = artworkDataJSONname + "_" + visitorDataCSV.name[:-4]
os.makedirs(currentPath + "/" + artworkVisitorDataDir, exist_ok=True)
# "Daegu_new_preAURA_mmdd_MMDD" => AURA 이전의 mmdd 부터 MMDD 까지의 Daegu_new 콜렉션의 작품들 .npy 들을 보관하는 폴더명

reader = csv.DictReader(visitorDataCSV)

xOffset = -2.4
yOffset = 0
# 현재 5전시실만 배열로 변환
process_visitor_data(
    heatmap, dictArray, visitorDataCSV, xOffset, yOffset, heatmapCellSize, ArtworkList
)

plt.figure(1)
sub_plots, axes = plt.subplots(1, 25, sharey=True)
num = 0

for artwork in ArtworkList:
    artwork_Np = create_empty_heatmap(spaceVertcalCells, spaceHorizontalCells)
    for yx, i in np.ndenumerate(dictArray):  # 나중에 수정 필요... 임시방편
        if artwork in dictArray[yx[0]][yx[1]]:
            artwork_Np[yx[0]][yx[1]] = int(dictArray[yx[0]][yx[1]][artwork])
    if np.max(artwork_Np) != 0:
        np.save(currentPath + "/" + artworkVisitorDataDir + "/" + artwork, artwork_Np)
        artwork_Np_CSV = pd.DataFrame(artwork_Np)
        sns.heatmap(artwork_Np_CSV, cmap="Greens", vmin=0, vmax=200, ax=axes[num])
        num += 1

heatmapCSV = pd.DataFrame(heatmap)
dictArrayCSV = pd.DataFrame(dictArray)
heatmapCSV.to_csv(visitorDataCSV.name[:-4] + "_Heatmap.csv", index=False)
dictArrayCSV.to_csv(visitorDataCSV.name[:-4] + "_DictArray.csv", index=False)

# sns.heatmap(heatmapCSV, cmap='Greens', vmin=0, vmax=200)
# plt.figure(2)
# sns.heatmap(h25, cmap='BuPu')
# plt.figure(3)
# sns.heatmap(h24, cmap='PuRd')

plt.show()
