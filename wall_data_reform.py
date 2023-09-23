import os
from time import localtime, time
import json
import math
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('wall_list_2023.pkl', 'rb') as f:
    wall_list = pickle.load(f)

for wall in wall_list:
    if(wall["id"] == "w1"):
        wall["displayable"] = False
    if(wall["id"] == "w21"):
        wall["displayable"] = False
    if(wall["id"] == "w36"):
        wall["displayable"] = False
    if(wall["id"] == "w38"):
        wall["displayable"] = False
    print(wall)

with open('wall_list_2023.pkl', 'wb') as f:
    pickle.dump(wall_list,f)