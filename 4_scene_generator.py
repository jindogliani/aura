"""
- 생성한 scene들은 .csv 파일 형태로 저장
- 3번에서 생성한 아트워크 영역 매트릭스 .npz로 공간 매트릭스를 훑으면서 scene 생성
"""

import copy
import csv
import json
import math
import os
from time import localtime, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize


cwd = os.getcwd()
artwork_data_path = "Daegu_new.json" #작품의 메타데이터가 있는 JSON 파일 => 작품의 size 값 추출
exhibition_data_path = "exhibition_2022.json" #작품이 걸려있는 전시 내용 JSON 파일 => 작품의 positions 값 추출

with open(artwork_data_path, "r") as f:
    artwork_data = json.load(f)
artwork_id_list = [{"id": artwork["id"], "size": artwork["dimensions"]} for artwork in artwork_data["exhibitionObjects"]] #작품 id 리스트

print(artwork_id_list.count)

with open(exhibition_data_path, "r") as f:
    exhibition_data = json.load(f)

for exhibited_artwork in exhibition_data["paintings"]:
    print(exhibited_artwork["position"]["x"])