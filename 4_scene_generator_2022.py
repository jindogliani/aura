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
import cv2
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
total_exhibited_artwork_list = [{"id": artwork["artworkIndex"], "pos_x": round(artwork["position"]["x"], 3), "pos_z": round(artwork["position"]["z"], 3)} for artwork in exhibition_data["paintings"]] #작품 id 리스트
#2022년 기준 13작품만 관람객 데이터를 수집 함
hall5_exhibited_artwork_list = ["PA-0064", "PA-0067", "PA-0027", "PA-0025", "PA-0087", "PA-0070", "PA-0066", "PA-0045", "PA-0079", "PA-0024", "PA-0085", "PA-0063", "PA-0083"]

exhibited_artwork_list =[] #2022년 하정웅미술관 5전시실 최종 13개 작품만. 2023년도 올해는 작품 개수 달라질 예정
for exhibited_artwork in total_exhibited_artwork_list:
    if exhibited_artwork["id"] in hall5_exhibited_artwork_list:
        exhibited_artwork_list.append(exhibited_artwork)

# print("Exhibited Artwork List in 2022")
for exhibited_artwork in exhibited_artwork_list:
    for artwork in artwork_list:
        if exhibited_artwork["id"] == artwork["id"]:
            exhibited_artwork["width"] = round(float(artwork["size"].split("x")[1]) / 100, 3) #width 정보만 빼오기
            # exhibited_artwork["height"] = round(float(artwork["size"].split("x")[0]) / 100, 3) #height 정보만 빼오기 #현재는 사용하지 않음.
            exhibited_artwork["placed"] = False
            # exhibited_artwork["size"] = round(float(artwork["size"].split("x")) / 100, 3) # split 잘 먹었는지 확인용
    for wall in wall_list: #TODO
        if wall["theta"] == 0: 
            if (wall["x1"] <= exhibited_artwork["pos_x"] <= wall["x2"]) and (abs(wall["z1"] - exhibited_artwork["pos_z"]) <= 0.2):
                exhibited_artwork["wall"] = wall["id"]
                exhibited_artwork["theta"] = wall["theta"]
        elif wall["theta"] == 180: 
            if (wall["x2"] <= exhibited_artwork["pos_x"] <= wall["x1"]) and (abs(wall["z1"] - exhibited_artwork["pos_z"]) <= 0.2):
                exhibited_artwork["wall"] = wall["id"]
                exhibited_artwork["theta"] = wall["theta"]        
        elif wall["theta"] == 90:
            if (wall["z1"] <= exhibited_artwork["pos_z"] <= wall["z2"]) and (abs(wall["x1"] - exhibited_artwork["pos_x"]) <= 0.2):
                exhibited_artwork["wall"] = wall["id"]
                exhibited_artwork["theta"] = wall["theta"]   
        elif wall["theta"] == -90:
            if (wall["z2"] <= exhibited_artwork["pos_z"] <= wall["z1"]) and (abs(wall["x1"] - exhibited_artwork["pos_x"]) <= 0.2):
                exhibited_artwork["wall"] = wall["id"]
                exhibited_artwork["theta"] = wall["theta"]   
        else:
            continue #TODO #작품에 오일러앵글 쓰자 ! 올해는!!
    # print(exhibited_artwork)
# print()
# print()

with open('exhibited_artwork_list_2022.pkl', 'wb') as f:
    pickle.dump(exhibited_artwork_list,f)

#씬생성 알고리즘
# 13! / (24 * 3600 * 10000)= 7.2일 걸린다...
#iteration 어떻게 할지 고민 필요... #TODO

space_vertical_size, space_horizontal_size = 20, 20   #공간 세로 길이: 20미터 | 공간 가로 길이: 20미터
heatmap_cell_size = 0.1   #히트맵 셀 사이즈: 0.1미터 = 10센티미터
space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size #히트맵 가로 셀 개수: 20/0.1 = 200개 | 히트맵 세로 셀 개수: 20/0.1 = 200개 
space_horizontal_cells, space_vertcal_cells = round(space_horizontal_cells), round(space_vertcal_cells)

space_heatmap = np.load('SpaceData/2022_coords_Ha5_ver2+(8-13).npy')
space_heatmap[space_heatmap > 0] = -10 #wall -10으로 표시해서 확인 용도

x_offset, z_offset = 4, 10
padding = 0.3 #기본적으로 작품 좌우 30cm의 여백을 줌


def heatmap_generator(
    artwork_width, new_pos_x, new_pos_z, old_pos_x, old_pos_z, x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap
):
    #작품 객체 indicate heatmap 생성
    x1, x2, z1, z2 = new_pos_x - artwork_width/2, new_pos_x + artwork_width/2, new_pos_z, new_pos_z
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_heatmap = np.zeros((200, 200), dtype = np.int16)
    artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #작품 프레임 두께 10cm
    
    #작품이 향하고 있는 방향 표시
    direction = np.zeros((200, 200), dtype = np.int16)
    direction = cv2.line(direction, (_x1, _z1), (round((_x1+_x2)/2), round((_z1+_z2)/2)), 255, 1)
    direction_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 90, 0.7)
    direction = cv2.warpAffine(direction, direction_rotation, direction.shape)
    artwork_heatmap += direction

    #작품 새로운 벽 방향으로 회전
    artwork_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 360-new_theta, 1)
    artwork_heatmap = cv2.warpAffine(artwork_heatmap, artwork_rotation, artwork_heatmap.shape)
    artwork_heatmap[artwork_heatmap > 0] = -20

    #작품 관람객 히트맵 회전 #TODO 맞는 회전을 위해 쎄타 값 여러번 조정 필요.. 
    artwork_visitor_transform = np.float32([[1, 0, old_pos_x - new_pos_x], [0, 1, old_pos_z - new_pos_z]])
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)
    artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta - old_theta, 1)
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)
    artwork_heatmap += artwork_visitor_heatmap

    return artwork_heatmap

start = time.time() #1회당 얼마 소요되는지 시간 체크
# _optimized_artwork_list = exhibited_artwork_list[:] #확인용1 - shallow copy는 안에 딕셔너리 내용이 수정되어서 무조건 deepcopy 해야 함..
optimized_artwork_list = copy.deepcopy(exhibited_artwork_list)
optimized_wall_list = copy.deepcopy(wall_list)
optimized_artwork_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16) #1px == 10cm 크기 

print("Wall List")
for wall in optimized_wall_list:
    wall["length"] = round(wall["length"], 2)
    wall["length_check"] = wall["length"]
    wall["hanged_artworks"] = []
    if wall["displayable"] == True: #벽이 현재 디스플레이가 가능한 상황인지 확인
        for optimized_artwork in optimized_artwork_list:
            if optimized_artwork["placed"] == False: #작품이 새로운 벽에 할당되었는지 확인
                # padding = math.sqrt(math.pow(exhibited_artwork["width"], 2) + math.pow(exhibited_artwork["height"], 2)) - exhibited_artwork["width"] #통상적으로 작품의 대각선 길이를 관람영역으로 함.
                if wall["length_check"] > (optimized_artwork["width"] + 2*padding): #작품 너비가 wall length_check 길이보다 작을 때
                    wall["length_check"] = wall["length_check"] - (optimized_artwork["width"] + 2*padding)
                    optimized_artwork["placed"] = True #작품 재배치 되었다고 표시
                    wall["hanged_artworks"].append(optimized_artwork) #이거 너무 복잡한데 구조가;; 원래는 optimized_artwork["id"]만 넣었음
                    # optimized_artwork["new_wall"] = wall["id"] #나중에 없애도 될 것 같음 #TODO
                    # optimized_artwork["new_theta"] = wall["theta"] #나중에 없애도 될 것 같음 #TODO
    wall["length_check"] = round(wall["length_check"], 2)
    if len(wall["hanged_artworks"]) != 0: #재배치 결과 작품이 벽에 할당되어 있을 때,
        if len(wall["hanged_artworks"]) == 1: #작품이 하나만 걸려 있을 때,
            wall["hanged_artworks"][0]["new_coords"] = ((wall["x1"] + wall["x2"])/2, (wall["z1"] + wall["z2"])/2) #그냥 벽 중앙에 고정
        else: #작품이 여러 개 벽에 할당되어 있을 때
            temp_len = 0
            temp_ratio = 0
            for hanged_artwork in wall["hanged_artworks"]:
                temp_len += hanged_artwork["width"]
            for i, hanged_artwork in enumerate(wall["hanged_artworks"], start=1):
                hanged_artwork["new_ratio"] = (round(temp_ratio, 3), round(temp_ratio+(hanged_artwork["width"] / temp_len), 3))
                new_coord_ratio = (hanged_artwork["new_ratio"][0] + hanged_artwork["new_ratio"][1])/2
                hanged_artwork["new_coords"] = (round(wall["x1"]*(1-new_coord_ratio) + wall["x2"]*new_coord_ratio, 3), round(wall["z1"]*(1-new_coord_ratio) + wall["z2"]*new_coord_ratio, 3))
                temp_ratio += hanged_artwork["width"] / temp_len
        for hanged_artwork in wall["hanged_artworks"]:
            artwork_visitor_heatmap = np.load('Daegu_new_preAURA_1025_1117+(8-13)/'+ hanged_artwork["id"] + '.npy')
            artwork_heatmap = heatmap_generator(hanged_artwork["width"], hanged_artwork["new_coords"][0], hanged_artwork["new_coords"][1], hanged_artwork["pos_x"], hanged_artwork["pos_z"], x_offset, z_offset, heatmap_cell_size, hanged_artwork["theta"], wall["theta"], artwork_visitor_heatmap)
            optimized_artwork_heatmap += artwork_heatmap
    print(wall)
    print()
    
optimized_artwork_heatmap += space_heatmap
# heatmapCSV = pd.DataFrame(optimized_artwork_heatmap)
# sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-20, vmax=10)
# plt.show()

end = time.time() #1회당 얼마 소요되는지 시간 체크

print()
print()
print((end - start))

