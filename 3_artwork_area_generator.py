"""
- 2번에서의 히트맵을 1번의 매트릭스랑 비교하여 작품들의 예상 영역 추출 후 매트릭스로 저장
- 잘 정합되었는지 확인용
"""
import os
from time import localtime, time

import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

currentPath = os.getcwd()
date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) + ')'

space_vertical_size, space_horizontal_size = 20, 20
heatmap_cell_size = 0.1
space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
space_horizontal_cells, space_vertcal_cells = round(space_horizontal_cells), round(space_vertcal_cells)

visitor_heatmap = np.load('Daegu_new_preAURA_1025_1117+(8-13)/preAURA_1025_1117_Heatmap+(8-13).npy')
space_heatmap = np.load('SpaceData/2022_coords_Ha5_ver2+(8-13).npy')
# space_heatmap[space_heatmap == 0] = -50
# space_heatmap[space_heatmap == 127] = 0 #wall -10으로 표시해서 확인 용도
space_heatmap[space_heatmap > 127] = -10 #wall -10으로 표시해서 확인 용도

with open('exhibited_artwork_list_2022.pkl', 'rb') as f: #전시 중인 13작품 사전 로드
    exhibited_artwork_list = pickle.load(f)

artwork_location_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16) #1px == 10cm 크기 
x_offset, z_offset = 4, 10

#작품들 위치 히트맵 위에 표시
for exhibited_artwork in exhibited_artwork_list:
    print(exhibited_artwork)
    x1, x2, z1, z2 = exhibited_artwork["pos_x"] - exhibited_artwork["width"]/2, exhibited_artwork["pos_x"] + exhibited_artwork["width"]/2, exhibited_artwork["pos_z"], exhibited_artwork["pos_z"]
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #작품 프레임 두께 10cm
    
    #작품이 향하고 있는 방향 표시
    direction = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    direction = cv2.line(direction, (_x1, _z1), (round((_x1+_x2)/2), round((_z1+_z2)/2)), 255, 1)
    direction_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 90, 0.7)
    direction = cv2.warpAffine(direction, direction_rotation, direction.shape)
    artwork_heatmap += direction

    #벽 방향으로 회전
    # print(exhibited_artwork["theta"])
    artwork_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), exhibited_artwork["_theta"], 1)
    artwork_heatmap = cv2.warpAffine(artwork_heatmap, artwork_rotation, artwork_heatmap.shape)
    
    # _artwork_heatmap = pd.DataFrame(artwork_heatmap)
    # sns.heatmap(_artwork_heatmap, cmap='RdYlGn_r', vmin=0, vmax=510)
    # plt.show()

    if(exhibited_artwork["id"] == "PA-0024"):
        artwork_visitor_heatmap = np.load('Daegu_new_preAURA_1025_1117+(8-13)/'+ exhibited_artwork["id"] + '.npy')
        
        artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 0, 1)
        artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)
        artwork_visitor_transform = np.float32([[1, 0, 0], [0, 1, 0]])
        artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)

        artwork_heatmap = artwork_heatmap + artwork_visitor_heatmap
        
        artwork_location_heatmap += artwork_heatmap
    

# _, artwork_location_heatmap = cv2.threshold(artwork_location_heatmap, 127, 255, cv2.THRESH_BINARY)
artwork_location_heatmap[artwork_location_heatmap > 250] = -30 #회화 작품 객체 -20으로 표시해서 확인 용도

heatmap = space_heatmap + artwork_location_heatmap
heatmapCSV = pd.DataFrame(heatmap)
sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-30, vmax=150)
plt.show()

