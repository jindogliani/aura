"""
2023/08/05
- 캡처 이미지 데이터를 공간 데이터로 활용하는데 어려움이 있음.
- 따라서 좌표 (coords) 데이터를 불러오는 방식으로 변경함.
- 2022년 하정웅미술관, 2023년 광주시립미술관 전체 데이터 사용
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

date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) +'-'+ str(localtime(time()).tm_hour) + '-'+ str(localtime(time()).tm_min) + ')'

def space_csv_to_heatmap(ver, spaceDataCSV, heatmap_cell_size, space_vertical_size, space_horizontal_size, x_offset, z_offset):
    
    spaceDataCSVname = ver + '_' + spaceDataCSV.name[10:-4]
    
    reader = csv.reader(spaceDataCSV)
    #walls=edges와 같은 말. 벽의 값을 갖고 오기 위해 좌표배열1, 2를 합침
    pointsArr= np.array(list(reader)) 
    points2Arr= copy.deepcopy(pointsArr) 
    points2Arr = np.roll(pointsArr, -1, axis=0)
    edgeArr = np.concatenate((pointsArr, points2Arr), axis=1)
    spaceDataCSV.close()

    spaceVertcalCells, spaceHorizontalCells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
    spaceHorizontalCells, spaceVertcalCells = round(spaceHorizontalCells), round(spaceVertcalCells)
    heatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint8)
    
    img = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint8)
    li = []

    for row in edgeArr:
        if(ver == '2023'):
            x1, x2, z1, z2 = -float(row[0]), -float(row[3]), float(row[2]), float(row[5])
        elif(ver == '2022'):
            x1, x2, z1, z2 = float(row[0]), float(row[3]), float(row[2]), float(row[5])
        _x1, _x2, _z1, _z2= int((x1+x_offset)/heatmap_cell_size), int((x2+x_offset)/heatmap_cell_size), int((z1+z_offset)/heatmap_cell_size), int((z2+z_offset)/heatmap_cell_size) #z축 뒤집히는 것은 유동적 수정 필요
        img = cv2.line(img, (_x1, _z1), (_x2, _z2), 255, 1)
        wallDic = dict()
        wallDic = {'id': '', 'displayable': True, 'length':0, 'theta':0, 'x1': 0, 'z1': 0, 'x2': 0,'z2': 0}
        wallDic['length']= math.dist((x1, z1), (x2, z2))
        wallDic['theta'] = abs(round(np.rad2deg(np.arctan2(z1 - z2, x1 - x2))) - 180) #TODO 2023
        
        if abs(wallDic['theta'] - 360) <= 5:
            wallDic['theta'] = 0
        
        wallDic['x1'], wallDic['x2'], wallDic['z1'], wallDic['z2'] = x1, x2, z1, z2
        li.append(wallDic)

    for i in range(len(li)):
        li[i]['id'] = 'w'+str(i)
        if li[i]['length'] <= 1.3:
            li[i]['displayable'] = False
        else:
            li[i]['displayable'] = True
        print(li[i])

    with open(ver + '_wall_list' + '.pkl', 'wb') as f:
        pickle.dump(li,f)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inside_img = cv2.drawContours(img.copy(), contours, -1, 127, thickness = -1)

    resized_img = cv2.resize(img, (int(spaceHorizontalCells), int(spaceVertcalCells))) #1px == 10cm
    resized_inside_img = cv2.resize(inside_img, (int(spaceHorizontalCells), int(spaceVertcalCells)))

    _, resized_img_th = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY)

    _resized_img_th = np.asarray(resized_img_th, dtype = np.int16)
    _resized_inside_img = np.asarray(resized_inside_img, dtype = np.int16)
    _resized_img_th = _resized_img_th + _resized_inside_img

    sMapCSV = pd.DataFrame(_resized_img_th)
    sns.heatmap(sMapCSV, cmap='Greens', vmin=0, vmax=255)
    plt.show()

    np.save("SpaceData/" + spaceDataCSVname + date, _resized_img_th)
    sMapCSV.to_csv("SpaceData/" + spaceDataCSVname + date + '_Heatmap.csv', index=False)


ver = "2022"

if ver == "2023":
    spaceDataCSV = open('SpaceData/GMA.csv', 'r', encoding='utf-8-sig')
    space_vertical_size, space_horizontal_size = 40, 40
    heatmap_cell_size = 0.1
    x_offset, z_offset = 7, 12 

    space_csv_to_heatmap(ver, spaceDataCSV, heatmap_cell_size, space_vertical_size, space_horizontal_size, x_offset, z_offset)
elif ver == "2022":
    spaceDataCSV = open('SpaceData/HJW.csv', 'r', encoding='utf-8-sig')
    space_vertical_size, space_horizontal_size = 40, 40
    heatmap_cell_size = 0.1
    x_offset, z_offset = 25, 20 

    space_csv_to_heatmap(ver, spaceDataCSV, heatmap_cell_size, space_vertical_size, space_horizontal_size, x_offset, z_offset)