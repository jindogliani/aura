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

image = cv2.imread("GalleryImage/GMA3_W-10.png", cv2.IMREAD_COLOR)
wImage = cv2.imread("GalleryImage/GMA3_W+0.5.png", cv2.IMREAD_COLOR)
imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
wImageGrey = cv2.cvtColor(wImage, cv2.COLOR_BGR2GRAY)

ret, imageThreshold = cv2.threshold(imageGrey, 254, 255, cv2.THRESH_BINARY)
ret, wImageThreshold = cv2.threshold(wImageGrey, 254, 255, cv2.THRESH_BINARY)
imageThreshold = cv2.bitwise_not(imageThreshold)
wImageThreshold = cv2.bitwise_not(wImageThreshold)

imageThreshold2 = cv2.adaptiveThreshold(imageGrey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
wImageThreshold2 = cv2.adaptiveThreshold(wImageGrey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

# canny = cv2.Canny(image, 254, 255)
# cv2.imshow("canny", canny)
# cContours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours, hierarchy = cv2.findContours(imageThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
wContours, hierarchy = cv2.findContours(wImageThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours + wContours

cntrs = []
for cnt in wContours:
    if cv2.contourArea(cnt) > 2000:
        cntrs.append(cnt)
        print(cv2.contourArea(cnt))

image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
 
cv2.imshow('Image', image)

# plt.figure(1)
# plt.imshow(cv2.cvtColor(wImageThreshold2, cv2.COLOR_GRAY2RGB))
# plt.figure(2)
# plt.imshow(cv2.cvtColor(imageThreshold, cv2.COLOR_GRAY2RGB))

plt.show()
cv2.waitKey(0)
cv2.imwrite("GalleryImage/test.png", imageThreshold)