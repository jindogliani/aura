"""
2023/08/05
- 캡처 이미지 데이터를 공간 데이터로 활용하는데 어려움이 있음.
- 따라서 좌표 (coords) 데이터를 불러오는 방식으로 변경함.
- 2023년 광주시립미술관 전체 데이터 사용
"""

import cv2
import numpy as np
import math
import csv
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import localtime, time
import os
import pickle

date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) + ')'

#공간 x,z 좌표 데이터를 읽어온다.
spaceDataCSV = open('SpaceData/coords_GMA3.csv', 'r', encoding='utf-8-sig') #2023년 광주시립미술관 전체 데이터
spaceDataCSVname = spaceDataCSV.name[10:-4]
reader = csv.reader(spaceDataCSV)

#walls=edges와 같은 말. 벽의 값을 갖고 오기 위해 좌표배열1, 2를 합침
pointsArr= np.array(list(reader)) 
points2Arr= copy.deepcopy(pointsArr) 
points2Arr = np.roll(pointsArr, -1, axis=0)
edgeArr = np.concatenate((pointsArr, points2Arr), axis=1)
spaceDataCSV.close()

#공간 세로 길이: 50미터 | 공간 가로 길이: 50미터
spaceVerticalSize, spaceHorizontalSize = 40, 30
#히트맵 셀 사이즈: 0.1미터 = 10센티미터
heatmapCellSize = 0.1

spaceVertcalCells, spaceHorizontalCells = spaceVerticalSize / heatmapCellSize, spaceHorizontalSize / heatmapCellSize
spaceHorizontalCells, spaceVertcalCells = round(spaceHorizontalCells), round(spaceVertcalCells)
#히트맵 가로 셀 개수: 10/0.2 = 50개 | 히트맵 세로 셀 개수: 20/0.2 = 100개 
heatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint8)
#히트맵 numpy 이중배열

img = np.zeros((spaceVerticalSize*10, spaceHorizontalSize*10), dtype = np.uint8) #1px == 10cm
li = []
xOffset, zOffset = 28, 28 #2022년 공간데이터와 관람객 데이터 사이의 위치 차이
#관람객 시작점이 상대좌표에서 (0, y, 0) 이었으나 절대좌표에서는 (-4.1, y, -5.8)

for row in edgeArr:
    x1, x2, z1, z2 = float(row[0]), float(row[3]), float(row[2]), float(row[5])
    _x1, _x2, _z1, _z2= int(10*(x1+xOffset)), int(10*(x2+xOffset)), int(10*(-z1+zOffset)), int(10*(-z2+zOffset)) #z축 뒤집히는 것은 유동적 수정 필요
    img = cv2.line(img, (_x1, _z1), (_x2, _z2), 255, 1)
    wallDic = dict()
    wallDic = {'id': '', 'displayable': True, 'length':0, 'theta':0, 'x1': 0, 'z1': 0, 'x2': 0,'z2': 0}
    wallDic['length'], wallDic['theta']= math.dist((x1, z1), (x2, z2)), np.rad2deg(np.arctan2(z2 - z1, x2 - x1))
    wallDic['_theta'] = abs(round(np.rad2deg(np.arctan2(z1 - z2, x1 - x2))) -180) #TODO
    wallDic['x1'], wallDic['x2'], wallDic['z1'], wallDic['z2'] = x1, x2, z1, z2
    li.append(wallDic)

for i in range(len(li)):
    li[i]['id'] = 'w'+str(i)
    if li[i]['length'] <= 1.3:
         li[i]['displayable'] = False
    else:
        li[i]['displayable'] = True
    print(li[i])

with open('wall_list_2023.pkl', 'wb') as f:
    pickle.dump(li,f)

#공간 데이터를 통해 생성한 img를 1/10으로 리사이즈함.
#이를 통해 10cm 크기의 cell을 갖는 공간 배열 생성
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
inside_img = cv2.drawContours(img.copy(), contours, -1, 127, thickness = -1)

resized_img = cv2.resize(img, (int(spaceHorizontalCells), int(spaceVertcalCells))) #1px == 10cm
resized_inside_img = cv2.resize(inside_img, (int(spaceHorizontalCells), int(spaceVertcalCells)))

_, resized_img_th = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY)

_resized_img_th = np.asarray(resized_img_th, dtype = np.int16)
_resized_inside_img = np.asarray(resized_inside_img, dtype = np.int16)
_resized_img_th = _resized_img_th + _resized_inside_img

cv2.imshow('th', resized_img_th)
cv2.imshow('image', img)
cv2.waitKey(0)

sMapCSV = pd.DataFrame(_resized_img_th)
sns.heatmap(sMapCSV, cmap='Greens', vmin=0, vmax=255)

plt.show()

np.save("SpaceData/" + spaceDataCSVname + date, _resized_img_th)
sMapCSV.to_csv("SpaceData/" + spaceDataCSVname + date + '_Heatmap.csv', index=False)
