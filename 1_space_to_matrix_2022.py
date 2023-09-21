"""
2023/08/05
- 캡처 이미지 데이터를 공간 데이터로 활용하는데 어려움이 있음.
- 따라서 좌표 (coords) 데이터를 불러오는 방식으로 변경함.
- 2022년 하정웅미술관 갤러리5 데이터 사용
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
spaceDataCSV = open('SpaceData/2022_coords_Ha5_ver2.csv', 'r', encoding='utf-8-sig') #2022년 하정웅미술관 갤러리5 데이터
spaceDataCSVname = spaceDataCSV.name[10:-4]
reader = csv.reader(spaceDataCSV)

#walls=edges와 같은 말. 벽의 값을 갖고 오기 위해 좌표배열1, 2를 합침
pointsArr= np.array(list(reader)) 
points2Arr= copy.deepcopy(pointsArr) 
points2Arr = np.roll(pointsArr, -1, axis=0)
edgeArr = np.concatenate((pointsArr, points2Arr), axis=1)
spaceDataCSV.close()

#공간 세로 길이: 20미터 | 공간 가로 길이: 20미터
spaceVerticalSize, spaceHorizontalSize = 20, 20
#히트맵 셀 사이즈: 0.2미터 = 20센티미터
heatmapCellSize = 0.1

#히트맵 가로 셀 개수: 20/0.1 = 200개 | 히트맵 세로 셀 개수: 20/0.1 = 200개 
spaceVertcalCells, spaceHorizontalCells = spaceVerticalSize / heatmapCellSize, spaceHorizontalSize / heatmapCellSize
spaceHorizontalCells, spaceVertcalCells = round(spaceHorizontalCells), round(spaceVertcalCells)

#히트맵 numpy 이중배열
heatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint8)
 
print(edgeArr) #벽(walls=edges) 좌표 확인 배열 print

img = np.zeros((spaceVerticalSize*100, spaceHorizontalSize*100), dtype = np.uint8) #1px == 1cm 크기 
li = [] #벽정보를 담는 리스트
xOffset, zOffset = 4, 10 #2022년 공간데이터를 매트릭스로 옮기기 위해 오프셋 값 지정
#관람객 시작점이 상대좌표에서 (0, y, 0) 이었으나 절대좌표에서는 (-4.1, y, -5.8)임.
#관람객 데이터의 좌표가 (x, y), 공간 데이터의 좌표가 (x', y') 라고 하면 x' = x - 4.1 이고 y' = y - 5.8임.
#근데 공간 데이터와 관람객 데이터를 모두 음수 처리 하여서 -y' = -y + 5.8이 됨.

for row in edgeArr:
    x1, x2, z1, z2 = float(row[0]), float(row[3]), float(row[2]), float(row[5])
    _x1, _x2, _z1, _z2= int(100*(x1+xOffset)), int(100*(x2+xOffset)), int(100*(z1+zOffset)), int(100*(z2+zOffset)) #z축 뒤집히는 것은 유동적 수정 필요
    img = cv2.line(img, (_x1, _z1), (_x2, _z2), 255, 8) #벽 두께 8cm
    wallDic = dict()
    wallDic = {'id': '', 'displayable': True, 'length':0, 'theta':0,  'x1': 0, 'z1': 0, 'x2': 0,'z2': 0}
    #wallDic['theta'] = round(np.rad2deg(np.arctan2(z2 - z1, x2 - x1))) #TODO #HTW
    wallDic['length']= round(math.dist((x1, z1), (x2, z2)),2)
    wallDic['theta'] = abs(round(np.rad2deg(np.arctan2(z1 - z2, x1 - x2))) -180) #TODO
    wallDic['x1'], wallDic['x2'], wallDic['z1'], wallDic['z2'] = x1, x2, z1, z2
    li.append(wallDic)

for i in range(len(li)):
    li[i]['id'] = 'w'+str(i)
    if li[i]['length'] <= 1.3: #TODO
         li[i]['displayable'] = False
    else:
        li[i]['displayable'] = True
    print(li[i])

with open('wall_list_2022.pkl', 'wb') as f:
    pickle.dump(li,f) #TODO

#공간 데이터를 통해 생성한 img를 1/10으로 리사이즈함.
#이를 통해 10cm 크기의 cell을 갖는 공간 배열 생성
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
inside_img = cv2.drawContours(img.copy(), contours, -1, 127, thickness = -1)

resized_img = cv2.resize(img, (int(spaceHorizontalCells), int(spaceVertcalCells))) #1px == 10cm 크기
resized_inside_img = cv2.resize(inside_img, (int(spaceHorizontalCells), int(spaceVertcalCells)))

_, resized_img_th = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY)

_resized_img_th = np.asarray(resized_img_th, dtype = np.int16)
_resized_inside_img = np.asarray(resized_inside_img, dtype = np.int16)
_resized_img_th = _resized_img_th + _resized_inside_img

# cv2.imshow('th', resized_img) # 1/10로 리사이즈한 공간 이미지
# cv2.imshow('image', img) #원본 크기의 공간 이미지
# cv2.waitKey(0)

spaceMatrixCSV = pd.DataFrame(_resized_img_th) #공간 배열 .csv 저장용 및 plot용 데이터 변환
sns.heatmap(spaceMatrixCSV, cmap='Greens', vmin=0, vmax=255)
plt.show()

np.save("SpaceData/" + spaceDataCSVname + date, _resized_img_th)
spaceMatrixCSV.to_csv("SpaceData/" + spaceDataCSVname + date + '_Heatmap.csv', index=False)
