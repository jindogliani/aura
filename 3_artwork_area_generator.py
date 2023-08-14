"""
- 2번에서의 히트맵을 1번의 매트릭스랑 비교하여 작품들의 예상 영역 추출 후 매트릭스로 저장
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
zero_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.uint16)

visitor_heatmap = np.load('Daegu_new_preAURA_1025_1117+(8-13)/preAURA_1025_1117_Heatmap+(8-13).npy')
# visitor_heatmap = np.load('Daegu_new_preAURA_1025_1117/preAURA_1025_1117+66.npy')
# _visitor_heatmap = np.concatenate((visitor_heatmap, zero_heatmap), axis=0) #6월 6일 npy는 200(cols)x100(rows)로 만들어져서 제로배열과 컨케러네이트 함.

with open('exhibited_artwork_list.pkl', 'rb') as f:
    exhibited_artwork_list = pickle.load(f)

artwork_location_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16) #1px == 10cm 크기 
x_offset, z_offset = 4, 10

#작품들 히트맵 위에 표시
for exhibited_artwork in exhibited_artwork_list:
    x1, x2, z1, z2 = exhibited_artwork["position_x"] - exhibited_artwork["width"]/2, exhibited_artwork["position_x"] + exhibited_artwork["width"]/2, exhibited_artwork["position_z"], exhibited_artwork["position_z"]
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_location_heatmap = cv2.line(artwork_location_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #벽 두께 10cm
    
_, artwork_location_heatmap = cv2.threshold(artwork_location_heatmap, 127, 255, cv2.THRESH_BINARY)
artwork_location_heatmap[artwork_location_heatmap > 0] = -20

space_heatmap = np.load('SpaceData/2022_coords_Ha5_ver2+(8-13).npy')
space_heatmap[space_heatmap > 0] = -10 #wall -100으로 표시해서 확인 용도

heatmap = space_heatmap + visitor_heatmap + artwork_location_heatmap
heatmapCSV = pd.DataFrame(heatmap)
sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-20, vmax=10)
plt.show()

