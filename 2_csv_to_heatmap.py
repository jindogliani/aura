"""
- 관람객 데이터 .csv파일을 불러와서 heatmap으로 표현
- artwork 영역을 매트릭스로 저장한 후에 .npy 혹은 .npz 형태로 저장
"""
import numpy as np
import pandas as pd
import copy
import csv
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize

"""
예상 공간 값으로 0값 들어가는 이중배열 생성
"""
spaceVerticalSize, spaceHorizontalSize = 10, 15
#공간 세로 길이: 30미터 | 공간 가로 길이: 60미터
heatmapCellSize = 0.2
#히트맵 셀 사이즈: 0.2미터 = 20센티미터
spaceVertcalCells, spaceHorizontalCells = spaceVerticalSize / heatmapCellSize, spaceHorizontalSize / heatmapCellSize
spaceHorizontalCells = round(spaceHorizontalCells)
spaceVertcalCells = round(spaceVertcalCells)
#히트맵 가로 셀 개수: 15/0.2 = 75개 | 히트맵 세로 셀 개수: 10/0.2 = 50개 
heatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint16)
#히트맵 numpy 이중배열

dic = dict()
#array1 = [dic]*spaceVertcalCells*spaceHorizontalCells
array1 = []
for i in range(spaceVertcalCells*spaceHorizontalCells):
        array1.append(copy.deepcopy(dic))
        i += 1
dictArray = np.reshape(array1, (spaceVertcalCells, spaceHorizontalCells))
#나중에 수정 필요... 임시방편

"""
작품별 npy 저장하는 폴더 생성 및 작품리스트 생성
"""
currentPath = os.getcwd()
visitorDataCSV = open('preAURA_1025_1030.csv', 'r')
with open('Daegu_new.json', 'r') as f:
        artworkDataJSON = json.load(f)
        artworkDataJSONname = f.name[:-5]
#관람객데이터 csv파일, 전시정보 json파일 open

ArtworkList = []
for x in artworkDataJSON['exhibitionObjects']:
        ArtworkList.append(x['id'])
#ArtworkList 리스트에 작품들 관리번호 호출

artworkVisitorDataDir = artworkDataJSONname + '_' + visitorDataCSV.name[:-4]
os.makedirs(currentPath+ "/" + artworkVisitorDataDir, exist_ok= True)
#"Daegu_new_preAURA_mmdd_MMDD" => AURA 이전의 mmdd 부터 MMDD 까지의 Daegu_new 콜렉션의 작품들 .npy 들을 보관하는 폴더명 

reader = csv.DictReader(visitorDataCSV)

xOffset = -2.4
yOffset = 0
#현재 5전시실만 배열로 변환

for line in reader: 
      newXcord = xOffset + float(line['move_x'])
      newYcord = yOffset + float(line['move_y'])
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
                          print('processing')
                          print(yx)      
print('end')
visitorDataCSV.close()

# PA25heatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = int)
# PA24heatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = int)

# for yx, i in np.ndenumerate(dictArray): #나중에 수정 필요... 임시방편
#       if 'PA-0025' in dictArray[yx[0]][yx[1]]:
#             PA25heatmap[yx[0]][yx[1]] = dictArray[yx[0]][yx[1]]['PA-0025']
#       if 'PA-0024' in dictArray[yx[0]][yx[1]]:
#             PA24heatmap[yx[0]][yx[1]] = dictArray[yx[0]][yx[1]]['PA-0024']

plt.figure(1)
sub_plots, axes = plt.subplots(1, 25, sharey=True)
num = 0

for artwork in ArtworkList:
      artworkNp = artwork + 'Heatmap'
      globals()[artworkNp] = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint16)
      artwork_Np = globals()[artworkNp]
      for yx, i in np.ndenumerate(dictArray): #나중에 수정 필요... 임시방편
            if artwork in dictArray[yx[0]][yx[1]]:
                  artwork_Np[yx[0]][yx[1]] = int(dictArray[yx[0]][yx[1]][artwork])
      if np.max(artwork_Np) !=0:
            np.save(currentPath + "/" +artworkVisitorDataDir + "/" + artwork, artwork_Np)
            artwork_Np_CSV = pd.DataFrame(artwork_Np)
            sns.heatmap(artwork_Np_CSV, cmap='Greens', vmin=0, vmax=200, ax=axes[num]) #히트맵 plot 띄우는 거 수정필요
            num += 1

heatmapCSV = pd.DataFrame(heatmap)
dictArrayCSV = pd.DataFrame(dictArray)
heatmapCSV.to_csv(visitorDataCSV.name[:-4] + '_Heatmap.csv', index=False)
dictArrayCSV.to_csv(visitorDataCSV.name[:-4] + '_DictArray.csv', index=False)

# sns.heatmap(heatmapCSV, cmap='Greens', vmin=0, vmax=200)
# plt.figure(2)
# sns.heatmap(h25, cmap='BuPu')
# plt.figure(3)
# sns.heatmap(h24, cmap='PuRd')

plt.show()