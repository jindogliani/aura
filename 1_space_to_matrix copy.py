#Unity 상에서 공간 캡처 데이터를 불러와서 매트릭스로 변형 및 .csv로 저장

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

date = '+' + str(localtime(time()).tm_mon) + str(localtime(time()).tm_mday)

#공간 x,z 좌표 데이터를 읽어온다.
spaceDataCSV = open('SpaceData/coords_GMA2.csv', 'r', encoding='utf-8-sig') #2022년 하정웅미술관 갤러리5 데이터
spaceDataCSVname = spaceDataCSV.name[10:-4]
reader = csv.reader(spaceDataCSV)

#walls=edges와 같은 말. 벽의 값을 갖고 오기 위해 좌표배열1, 2를 합침
pointsArr= np.array(list(reader)) 
points2Arr= copy.deepcopy(pointsArr) 
points2Arr = np.roll(pointsArr, -1, axis=0)
edgeArr = np.concatenate((pointsArr, points2Arr), axis=1)
spaceDataCSV.close()

spaceVerticalSize, spaceHorizontalSize = 50, 50
#공간 세로 길이: 10미터 | 공간 가로 길이: 20미터
heatmapCellSize = 0.2
#히트맵 셀 사이즈: 0.2미터 = 20센티미터
spaceVertcalCells, spaceHorizontalCells = spaceVerticalSize / heatmapCellSize, spaceHorizontalSize / heatmapCellSize
spaceHorizontalCells, spaceVertcalCells = round(spaceHorizontalCells), round(spaceVertcalCells)
#히트맵 가로 셀 개수: 10/0.2 = 50개 | 히트맵 세로 셀 개수: 20/0.2 = 100개 
heatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint8)
#히트맵 numpy 이중배열

img = np.zeros((spaceVerticalSize*10, spaceHorizontalSize*10), dtype = np.uint8) #1px == 10cm
li = []
xOffset, zOffset = 37, 35 #2022년 공간데이터와 관람객 데이터 사이의 위치 차이
#관람객 시작점이 상대좌표에서 (0, y, 0) 이었으나 절대좌표에서는 (-4.1, y, -5.8)

for row in edgeArr:
    x1, x2, z1, z2 = float(row[0]), float(row[3]), float(row[2]), float(row[5])
    _x1, _x2, _z1, _z2= int(10*(x1+xOffset)), int(10*(x2+xOffset)), int(10*(-z1+zOffset)), int(10*(-z2+zOffset)) #z축 뒤집히는 것은 유동적 수정 필요
    img = cv2.line(img, (_x1, _z1), (_x2, _z2), 255, 1)
    wallDic = dict()
    wallDic = {'id': '', 'displayable': True, 'length':0, 'theta':0, 'x1': 0, 'z1': 0, 'x2': 0,'z2': 0}
    wallDic['length'], wallDic['theta']= math.dist((x1, z1), (x2, z2)), np.rad2deg(np.arctan2(z2 - z1, x2 - x1))
    wallDic['x1'], wallDic['x2'], wallDic['z1'], wallDic['z2'] = x1, x2, z1, z2
    li.append(wallDic)

for i in range(len(li)):
    li[i]['id'] = 'w'+str(i)
    if li[i]['length'] < 1.5:
         li[i]['displayable'] = False
    else:
        li[i]['displayable'] = True
    print(li[i])

resized_img = cv2.resize(img, (int(spaceHorizontalCells)*2, int(spaceVertcalCells)*2)) #1px == 10cm
_, resized_img_th = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY)
_resized_img_th = np.asarray(resized_img_th, dtype = np.int8)

cv2.imshow('th', resized_img_th)
cv2.imshow('image', img)

sMapCSV = pd.DataFrame(resized_img_th)
sns.heatmap(sMapCSV, cmap='Greens', vmin=0, vmax=1)

cv2.waitKey(0)
plt.show()

np.save("SpaceData/" + spaceDataCSVname + date, _resized_img_th)
sMapCSV.to_csv("SpaceData/" + spaceDataCSVname + date + '_Heatmap.csv', index=False)

'''
지금 numpy iter가 굉장히 마음에 들지 않는데, 그냥 사용 중.. 나중에 대체 방법 확인 필요
it = np.nditer(edgeArr, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index
    print(idx[0], idx[1], edgeArr[idx])
    it.iternext()
'''



