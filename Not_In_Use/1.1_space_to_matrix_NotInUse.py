#Unity 상에서 공간 캡처 데이터를 불러와서 매트릭스로 변형 및 .csv로 저장

'''
!!!NOT IN USE!!!

현재 해당 방식은 사용하지 않는 걸로
'''

import cv2
import numpy as np
import networkx as nx
import json 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from time import sleep
import os

currentPath = os.getcwd()

# GMA_W-10 는 갤러리 기준 -10미터 위에 white plane이 있을 때 캡처: 전체 갤러리 탑뷰
# GMA_W+0.5 는 갤러리 기준 0.5미터 위에 white plane이 있을 때 캡처: 갤러리 내부 가벽 탑뷰
image = cv2.imread("GalleryImage/GMA3_W-10.png", cv2.IMREAD_COLOR)
wImage = cv2.imread("GalleryImage/GMA3_W+0.5.png", cv2.IMREAD_COLOR)
# grey scale로 색상 변환
imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
wImageGrey = cv2.cvtColor(wImage, cv2.COLOR_BGR2GRAY)
# 흰색 제외 전부 날리기
_, imageThreshold = cv2.threshold(imageGrey, 254, 255, cv2.THRESH_BINARY)
_, wImageThreshold = cv2.threshold(wImageGrey, 254, 255, cv2.THRESH_BINARY)
# 색상 반전
imageThreshold = cv2.bitwise_not(imageThreshold)
wImageThreshold = cv2.bitwise_not(wImageThreshold)

'''
!!!NOT IN USE!!!

!!!ABSOLUTELY NOT IN USE!!!
imageThreshold2 = cv2.adaptiveThreshold(imageGrey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
wImageThreshold2 = cv2.adaptiveThreshold(wImageGrey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
canny = cv2.Canny(image, 254, 255)
cv2.imshow("canny", canny)
cContours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
'''

# 컨투어 생성
contours, hierarchy = cv2.findContours(imageThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
wContours, hierarchy = cv2.findContours(wImageThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours + wContours

# 내벽 컨투어 검출 => 근데 컨투어에 문제가 있는 상황..
cntrs = []
for cnt in wContours:
    if cv2.contourArea(cnt) > 2000:
        cntrs.append(cnt)


mask = np.zeros(imageGrey.shape, np.uint8)
cv2.drawContours(mask, contours, -1, 255, 4)
pixel_cv = cv2.findNonZero(mask)
print(type(pixel_cv))
print(pixel_cv)
print(pixel_cv[0][0])
galleryH = int(imageGrey.shape[0]*0.045)
galleryW = int(imageGrey.shape[1]*0.045)
heatmapCellSize = 0.2
hCells, wCells = galleryH / heatmapCellSize, galleryW / heatmapCellSize

hMap = np.zeros((int(hCells), int(wCells)), np.uint8)
print(type(hMap))
print(hMap)


img = image.copy()
# cv2.drawContours(img, [pixel_cv], -1, (0, 0, 255), 1)
cv2.imshow('numpy', mask)
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
resized_mask = cv2.resize(mask, (int(wCells), int(hCells)))
_, add_th = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("th", add_th)

hMapCSV = pd.DataFrame(add_th)
sns.heatmap(add_th, cmap='Greens', vmin=0, vmax=1)
 
print(resized_mask.shape)

cv2.imshow("resized", resized_mask)
#픽셀 1050 갤러리 너비 48미터 => 1픽셀 당 0.045미터 정도...

cv2.imshow('Image', image)
# plt.figure(1)
# plt.imshow(cv2.cvtColor(wImageThreshold, cv2.COLOR_GRAY2RGB))
# plt.figure(2)
# plt.imshow(cv2.cvtColor(imageThreshold, cv2.COLOR_GRAY2RGB))
plt.show()
cv2.waitKey(0)
cv2.imwrite("GalleryImage/test.png", imageThreshold)