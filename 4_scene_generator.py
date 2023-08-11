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

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize

cwd = os.getcwd()
artwork_data_path = "Daegu_new.json" #작품의 메타데이터가 있는 JSON 파일 => 작품의 size 값 추출
exhibition_data_path = "Data_2022.json" #작품이 걸려있는 전시 내용 JSON 파일 => 작품의 positions 값 추출

#전체 작품 111개 리스트
with open(artwork_data_path, "r") as f:
    artwork_data = json.load(f)
artwork_list = [{"id": artwork["id"], "size": artwork["dimensions"]} for artwork in artwork_data["exhibitionObjects"]] #작품 id 리스트

#전시된 작품 40개 리스트
with open(exhibition_data_path, 'r', -1, encoding='utf-8') as f:
    exhibition_data = json.load(f)
exhibited_artwork_list = [{"id": artwork["artworkIndex"], "position_x": artwork["position"]["x"], "position_z": artwork["position"]["z"]} for artwork in exhibition_data["paintings"]] #작품 id 리스트

#전시된 작품 리스트 사전에 사이즈 추가 => width로 수정 필요 #TODO
for exhibited_artwork in exhibited_artwork_list:
    for artwork in artwork_list:
        if exhibited_artwork["id"] == artwork["id"]:
            exhibited_artwork["width"] = float(artwork["size"].split("x")[0]) / 100 #width 정보만 빼오기
            exhibited_artwork["height"] = float(artwork["size"].split("x")[1]) / 100 #height 정보만 빼오기
            exhibited_artwork["displayed"] = False
            # exhibited_artwork["size"] = artwork["size"].split("x") #확인용
            # exhibited_artwork["dimension"] = artwork["size"] #확인용2
    print(exhibited_artwork)

print("total artwork count: " + str(len(artwork_list)))
print("exhibited artwork count: " + str(len(exhibited_artwork_list)))

#2022년도 하정웅 미술관 벽 정보 리스트 로드
with open('wall_list.pkl', 'rb') as f:
    wall_list = pickle.load(f)

# for wall in wall_list:
#     print(wall)

#씬생성 알고리즘

padding = 0.3 #기본적으로 작품 좌우 30cm의 여백을 줌

# 13! / (24 * 3600 * 10000)= 7.2일 걸린다...

for wall in wall_list:
    wall["displayed_artworks"] = []
    if wall["displayable"] == True: #벽이 현재 디스플레이가 가능한 상황인지 확인
        for exhibited_artwork in exhibited_artwork_list:
            if exhibited_artwork["displayed"] == False: #작품이 배치되었는지 확인. 배치가 안되었다면 배치
                if wall["length"] > exhibited_artwork["width"]: #너비가 벽 길이보다 작을 때
                    padding = math.sqrt(math.pow(exhibited_artwork["width"], 2) + math.pow(exhibited_artwork["height"], 2)) #통상적으로 작품의 대각선 길이를 관람영역으로 함.
                    wall["length"] = wall["length"] - (exhibited_artwork["width"] + 2*(padding - exhibited_artwork["width"]))
                    exhibited_artwork["displayed"] = True #작품 배치 되었다고 표시
                    wall["displayed_artworks"].append(exhibited_artwork["id"])
                    if wall["length"] < 0:
                        wall["length"] = 0
    print(wall)


