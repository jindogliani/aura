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
with open('wall_list_2022.pkl', 'rb') as f:
    wall_list = pickle.load(f)

#전체 작품 111개 리스트
with open(artwork_data_path, "r") as f:
    artwork_data = json.load(f)
artwork_list = [{"id": artwork["id"], "size": artwork["dimensions"]} for artwork in artwork_data["exhibitionObjects"]] #작품 id 리스트
#전시된 작품 40개 리스트
with open(exhibition_data_path, 'r', -1, encoding='utf-8') as f:
    exhibition_data = json.load(f)
total_exhibited_artwork_list = [{"id": artwork["artworkIndex"], "position_x": round(artwork["position"]["x"], 3), "position_z": round(artwork["position"]["z"], 3)} for artwork in exhibition_data["paintings"]] #작품 id 리스트
#2022년 기준 13작품만 관람객 데이터를 수집 함
hall5_exhibited_artwork_list = ["PA-0064", "PA-0067", "PA-0027", "PA-0025", "PA-0087", "PA-0070", "PA-0066", "PA-0045", "PA-0079", "PA-0024", "PA-0085", "PA-0063", "PA-0083"]

exhibited_artwork_list =[] #2022년 하정웅미술관 5전시실 최종 13개 작품만. 2023년도 올해는 작품 개수 달라질 예정
for exhibited_artwork in total_exhibited_artwork_list:
    if exhibited_artwork["id"] in hall5_exhibited_artwork_list:
        exhibited_artwork_list.append(exhibited_artwork)

for exhibited_artwork in exhibited_artwork_list:
    for artwork in artwork_list:
        if exhibited_artwork["id"] == artwork["id"]:
            exhibited_artwork["width"] = round(float(artwork["size"].split("x")[1]) / 100, 3) #width 정보만 빼오기
            exhibited_artwork["height"] = round(float(artwork["size"].split("x")[0]) / 100, 3) #height 정보만 빼오기
            exhibited_artwork["placed"] = False
            # exhibited_artwork["dimension"] = artwork["size"] # split 잘 먹었는지 확인용
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
    # print(exhibited_artwork)

with open('exhibited_artwork_list_2022.pkl', 'wb') as f:
    pickle.dump(exhibited_artwork_list,f)

#씬생성 알고리즘
# 13! / (24 * 3600 * 10000)= 7.2일 걸린다...
#iteration 어떻게 할지 고민 필요... #TODO

padding = 0.3 #기본적으로 작품 좌우 30cm의 여백을 줌

start = time.time() #1회당 얼마 소요되는지 시간 체크

# _optimized_artwork_list = exhibited_artwork_list[:] #확인용1 - shallow copy는 안에 딕셔너리 내용이 수정되어서 무조건 deepcopy 해야 함..
optimized_artwork_list = copy.deepcopy(exhibited_artwork_list)
optimized_wall_list = copy.deepcopy(wall_list)

#공간 정보 (벽) 딕셔너리
for wall in optimized_wall_list:
    wall["length"] = round(wall["length"], 2)
    wall["length_check"] = wall["length"]
    wall["displayed_artworks"] = []
    if wall["displayable"] == True: #벽이 현재 디스플레이가 가능한 상황인지 확인
        for optimized_artwork in optimized_artwork_list:
            if optimized_artwork["placed"] == False: #작품이 새로운 벽에 할당되었는지 확인
                if wall["length_check"] > optimized_artwork["width"]: #작품 너비가 wall length_check 길이보다 작을 때
                    # padding = math.sqrt(math.pow(exhibited_artwork["width"], 2) + math.pow(exhibited_artwork["height"], 2)) - exhibited_artwork["width"] #통상적으로 작품의 대각선 길이를 관람영역으로 함.
                    wall["length_check"] = wall["length_check"] - (optimized_artwork["width"] + 2*padding)
                    optimized_artwork["placed"] = True #작품 재배치 되었다고 표시
                    optimized_artwork["new_wall"] = wall["id"]
                    optimized_artwork["new_theta"] = wall["theta"]
                    wall["displayed_artworks"].append(optimized_artwork["id"])
        for i, displayed_artwork in enumerate(wall["displayed_artworks"], start=1):
            continue
    if wall["length_check"] < 0:
        wall["length_check"] = 0
    wall["length_check"] = round(wall["length_check"], 2)


    print(wall)

print()
print()
print("Optimized Artwork List")
for exhibited_artwork in optimized_artwork_list:
    print(exhibited_artwork)

end = time.time() #1회당 얼마 소요되는지 시간 체크

print((end - start)*math.factorial(13) /(3600*24))

