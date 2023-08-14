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
import time

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

#2022년도 하정웅 미술관 벽 정보 리스트 로드
with open('wall_list.pkl', 'rb') as f:
    wall_list = pickle.load(f)

#전체 작품 111개 리스트
with open(artwork_data_path, "r") as f:
    artwork_data = json.load(f)
artwork_list = [{"id": artwork["id"], "size": artwork["dimensions"]} for artwork in artwork_data["exhibitionObjects"]] #작품 id 리스트
#전시된 작품 40개 리스트
with open(exhibition_data_path, 'r', -1, encoding='utf-8') as f:
    exhibition_data = json.load(f)
total_exhibited_artwork_list = [{"id": artwork["artworkIndex"], "position_x": artwork["position"]["x"], "position_z": artwork["position"]["z"]} for artwork in exhibition_data["paintings"]] #작품 id 리스트
#2022년 기준 13작품만 관람객 데이터를 수집 함
hall5_exhibited_artwork_list = ["PA-0064", "PA-0067", "PA-0027", "PA-0025", "PA-0087", "PA-0070", "PA-0066", "PA-0045", "PA-0079", "PA-0024", "PA-0085", "PA-0063", "PA-0083"]

exhibited_artwork_list =[] #최종 13개 작품만. 2023년도 올해는 작품 개수 달라질 수 있음!
for exhibited_artwork in total_exhibited_artwork_list:
    if exhibited_artwork["id"] in hall5_exhibited_artwork_list:
        exhibited_artwork_list.append(exhibited_artwork)

for exhibited_artwork in exhibited_artwork_list:
    for artwork in artwork_list:
        if exhibited_artwork["id"] == artwork["id"]:
            exhibited_artwork["width"] = float(artwork["size"].split("x")[1]) / 100 #width 정보만 빼오기
            exhibited_artwork["height"] = float(artwork["size"].split("x")[0]) / 100 #height 정보만 빼오기
            exhibited_artwork["placed"] = False
            # exhibited_artwork["size"] = artwork["size"].split("x") #확인용
            # exhibited_artwork["dimension"] = artwork["size"] #확인용2
    for wall in wall_list: #TODO
        if wall["theta"] == 0: 
            if (wall["x1"] <= exhibited_artwork["position_x"] <= wall["x2"]) and (abs(wall["z1"] - exhibited_artwork["position_z"]) <= 0.2):
                exhibited_artwork["wall"] = wall["id"]
                exhibited_artwork["theta"] = wall["theta"]
        elif wall["theta"] == 180: 
            if (wall["x2"] <= exhibited_artwork["position_x"] <= wall["x1"]) and (abs(wall["z1"] - exhibited_artwork["position_z"]) <= 0.2):
                exhibited_artwork["wall"] = wall["id"]
                exhibited_artwork["theta"] = wall["theta"]        
        elif wall["theta"] == 90:
            if (wall["z1"] <= exhibited_artwork["position_z"] <= wall["z2"]) and (abs(wall["x1"] - exhibited_artwork["position_x"]) <= 0.2):
                exhibited_artwork["wall"] = wall["id"]
                exhibited_artwork["theta"] = wall["theta"]   
        elif wall["theta"] == -90:
            if (wall["z2"] <= exhibited_artwork["position_z"] <= wall["z1"]) and (abs(wall["x1"] - exhibited_artwork["position_x"]) <= 0.2):
                exhibited_artwork["wall"] = wall["id"]
                exhibited_artwork["theta"] = wall["theta"]   
        else:
            continue #TODO
    print(exhibited_artwork)

print("total artwork count: " + str(len(artwork_list)))
print("exhibited artwork count: " + str(len(exhibited_artwork_list)))

with open('exhibited_artwork_list.pkl', 'wb') as f:
    pickle.dump(exhibited_artwork_list,f)

#씬생성 알고리즘
# 13! / (24 * 3600 * 10000)= 7.2일 걸린다...
#iteration 어떻게 할지 고민 필요... #TODO

padding = 0.3 #기본적으로 작품 좌우 30cm의 여백을 줌

start = time.time()
for wall in wall_list:
    wall["displayed_artworks"] = []
    if wall["displayable"] == True: #벽이 현재 디스플레이가 가능한 상황인지 확인
        for exhibited_artwork in exhibited_artwork_list:
            if exhibited_artwork["placed"] == False: #작품이 재배치되었는지 확인. 재배치가 안되었다면 재배치
                if wall["length"] > exhibited_artwork["width"]: #너비가 벽 길이보다 작을 때
                    # padding = math.sqrt(math.pow(exhibited_artwork["width"], 2) + math.pow(exhibited_artwork["height"], 2)) - exhibited_artwork["width"] #통상적으로 작품의 대각선 길이를 관람영역으로 함.
                    wall["length"] = wall["length"] - (exhibited_artwork["width"] + 2*padding)
                    exhibited_artwork["placed"] = True #작품 재배치 되었다고 표시
                    wall["displayed_artworks"].append(exhibited_artwork["id"])
                    if wall["length"] < 0:
                        wall["length"] = 0
    print(wall)



end = time.time()

print((end - start)*math.factorial(13) /(3600*24))