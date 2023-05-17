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

print('start')
image = cv2.imread("GalleryImage/GMA.png", cv2.IMREAD_COLOR)
imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, imageThreshold = cv2.threshold(imageGrey, 106, 255, cv2.THRESH_BINARY)
imageThreshold2 = cv2.adaptiveThreshold(imageGrey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
print('end')

# cv2.imshow('GMA', imageGrey)
plt.figure(1)
plt.imshow(cv2.cvtColor(imageThreshold2, cv2.COLOR_GRAY2RGB))
plt.figure(2)
plt.imshow(cv2.cvtColor(imageThreshold, cv2.COLOR_GRAY2RGB))
plt.show()
cv2.waitKey(0)
cv2.imwrite("GalleryImage/test.png", imageThreshold)