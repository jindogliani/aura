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
date = '+' + str(localtime(time()).tm_mon) + str(localtime(time()).tm_mday)

spaceVerticalSize, spaceHorizontalSize = 10, 20
heatmapCellSize = 0.1
spaceVertcalCells, spaceHorizontalCells = spaceVerticalSize / heatmapCellSize, spaceHorizontalSize / heatmapCellSize
spaceHorizontalCells, spaceVertcalCells = round(spaceHorizontalCells), round(spaceVertcalCells)
zeroHeatmap = np.zeros((spaceVertcalCells, spaceHorizontalCells), dtype = np.uint16)

visitorHeatmap = np.load('Daegu_new_preAURA_1025_1030/preAURA_1025_1030+66.npy')
spaceHeatmap = np.load('SpaceData/2022_coords_Ha5+66.npy')

_visitorHeatmap = np.concatenate((visitorHeatmap, zeroHeatmap), axis=0)
spaceHeatmap[spaceHeatmap > 0] = -100 

heatmap = spaceHeatmap + _visitorHeatmap
heatmapCSV = pd.DataFrame(heatmap)
sns.heatmap(heatmapCSV, cmap='Greens', vmin=-100, vmax=200)
plt.show()
print("현재 위치:" + currentPath)

