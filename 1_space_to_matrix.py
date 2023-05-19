#Unity 상에서 공간 캡처 데이터를 불러와서 매트릭스로 변형 및 .csv로 저장

import cv2
import numpy as np
import networkx as nx
import json 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from time import sleep
import os

currentPath = os.getcwd()

#GMA_W-10 는 갤러리 기준 -10미터 위에 plane이 
#GMA_W+0.5 는 갤러리 기준 0.5미터 위에 plane이 캡처를 진행한
image = cv2.imread("GalleryImage/GMA3_W-10.png", cv2.IMREAD_COLOR)
wImage = cv2.imread("GalleryImage/GMA3_W+0.5.png", cv2.IMREAD_COLOR)

#회색으로 색상 변환
imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
wImageGrey = cv2.cvtColor(wImage, cv2.COLOR_BGR2GRAY)

#
ret, imageThreshold = cv2.threshold(imageGrey, 254, 255, cv2.THRESH_BINARY)
ret, wImageThreshold = cv2.threshold(wImageGrey, 254, 255, cv2.THRESH_BINARY)

# 색상 반전
imageThreshold = cv2.bitwise_not(imageThreshold)
wImageThreshold = cv2.bitwise_not(wImageThreshold)

# 사용하지 않음
# imageThreshold2 = cv2.adaptiveThreshold(imageGrey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
# wImageThreshold2 = cv2.adaptiveThreshold(wImageGrey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
# canny = cv2.Canny(image, 254, 255)
# cv2.imshow("canny", canny)
# cContours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어 생성
contours, hierarchy = cv2.findContours(imageThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
wContours, hierarchy = cv2.findContours(wImageThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours + wContours

cntrs = []
for cnt in wContours:
    if cv2.contourArea(cnt) > 3000:
        cntrs.append(cnt)

c = cntrs[0]
mask = np.zeros(imageGrey.shape, np.uint8)
cv2.drawContours(mask, [c], -1, 255, 2)

pixel_np = np.transpose(np.nonzero(mask))
pixel_cv = cv2.findNonZero(mask)
img = image.copy()
cv2.drawContours(img, [pixel_cv], -1, (0, 0, 255), 1)
cv2.imshow('numpy', mask)

image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# cv2.imshow('Image', image)

# plt.figure(1)
# plt.imshow(cv2.cvtColor(wImageThreshold2, cv2.COLOR_GRAY2RGB))
# plt.figure(2)
# plt.imshow(cv2.cvtColor(imageThreshold, cv2.COLOR_GRAY2RGB))

plt.show()
cv2.waitKey(0)
cv2.imwrite("GalleryImage/test.png", imageThreshold)