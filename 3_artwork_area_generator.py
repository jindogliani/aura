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

visitor_heatmap = np.load('Daegu_new_preAURA_1025_1030/preAURA_1025_1030+66.npy')
_visitor_heatmap = np.concatenate((visitor_heatmap, zero_heatmap), axis=0) #6월 6일 npy는 200(cols)x100(rows)로 만들어져서 제로배열과 컨케러네이트 함.

space_heatmap = np.load('SpaceData/2022_coords_Ha5+66.npy')
space_heatmap[space_heatmap > 0] = -100 #wall -100으로 표시해서 확인 용도

heatmap = space_heatmap + _visitor_heatmap
heatmapCSV = pd.DataFrame(heatmap)
sns.heatmap(heatmapCSV, cmap='Greens', vmin=-100, vmax=200)
plt.show()

