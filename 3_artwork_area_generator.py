"""
- 2번에서의 히트맵을 1번의 매트릭스랑 비교하여 작품들의 예상 영역 추출 후 매트릭스로 저장
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import localtime, time

currentPath = os.getcwd()
date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) + ')'

space_vertical_size, space_horizontal_size = 10, 20
heatmap_cell_size = 0.1
space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
space_horizontal_cells, space_vertcal_cells = round(space_horizontal_cells), round(space_vertcal_cells)
zero_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.uint16)

# visitor_heatmap = np.load('Daegu_new_preAURA_1025_1117/preAURA_1025_1117+66.npy')
visitor_heatmap = np.load('Daegu_new_preAURA_1025_1117+(8-13)/preAURA_1025_1117_Heatmap+(8-13).npy')
_visitor_heatmap = np.concatenate((visitor_heatmap, zero_heatmap), axis=0) #6월 6일 npy는 200(cols)x100(rows)로 만들어져서 제로배열과 컨케러네이트 함.

# space_heatmap = np.load('SpaceData/2022_coords_Ha5+66.npy')
space_heatmap = np.load('SpaceData/2022_coords_Ha5_ver2+(8-13).npy')
space_heatmap[space_heatmap > 0] = -10 #wall -100으로 표시해서 확인 용도

heatmap = space_heatmap + visitor_heatmap
heatmapCSV = pd.DataFrame(heatmap)
sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-10, vmax=10)
plt.show()

