"""
- 2번에서의 히트맵을 1번의 매트릭스랑 비교하여 작품들의 예상 영역 추출 후 매트릭스로 저장
- 잘 정합되었는지 확인용
"""
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

currentPath = os.getcwd()
date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) + ')'

space_vertical_size, space_horizontal_size = 40, 40
heatmap_cell_size = 0.1
space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
space_horizontal_cells, space_vertcal_cells = round(space_horizontal_cells), round(space_vertcal_cells)

#2022년도 하정웅 미술관 벽 정보 리스트 로드
with open('wall_list_2023.pkl', 'rb') as f:
    wall_list = pickle.load(f)

space_heatmap = np.load('SpaceData/coords_GMA3+(9-20).npy')
space_heatmap[space_heatmap > 254] = -6 #공간 벽을 -10으로 변환
space_heatmap[space_heatmap == 0] = -3 #공간 외부 값을 0에서 -15으로 전환
space_heatmap[space_heatmap == 127] = 0 #공간 내부 값을 127에서 0으로 전환

artwork_data_path = "Daegu_new.json" #작품의 메타데이터가 있는 JSON 파일 => 작품의 size 값 추출
exhibition_data_path = "Data_2023.json" #작품이 걸려있는 전시 내용 JSON 파일 => 작품의 positions 값 추출

#전체 작품 111개 리스트
with open(artwork_data_path, "r") as f:
    artwork_data = json.load(f)

exhibited_artwork_order = ["PA-0023", "PA-0026", "KO-0009", "PA-0095", "PA-0098", "PA-0076", "PA-0074", "PA-0075", "PA-0077", "KO-0010", "KO-0008", "PA-0101", "PA-0057", "PA-0052", "PA-0061", "PA-0001", "PA-0003", "PA-0004", "PA-0082", "PA-0084", "PA-0083", "PA-0063", "PA-0067", "PA-0064", "PA-0024", "PA-0087", "PA-0027", "PA-0025", "PA-0036", "PA-0085", "PA-0086", "PA-0070", "PA-0065", "PA-0031", "PA-0088", "PA-0100", "PA-0099", "KO-0007", "KO-0006", "KO-0004", "KO-0005", "PA-0090", "PA-0089"]

artwork_list = [{"id": artwork["id"], "size": artwork["dimensions"], "artist": artwork["artists"][0]} for artwork in artwork_data["exhibitionObjects"]] #작품 id 리스트

#전시된 작품 43개 리스트
exhibited_artwork_list =[]
with open(exhibition_data_path, 'r', -1, encoding='utf-8') as f:
    exhibition_data = json.load(f)

# for order in exhibited_artwork_order:
#     for artwork in exhibition_data["paintings"]:
#         if artwork["id"] == order:
#             exhibition_data["_paintings"].append(artwork)

# with open("Data_2023_ordered", "w") as f:
#     json.dump(exhibition_data, f)

exhibited_artwork_list = [{"id": artwork["artworkIndex"], "wall": "-", "pos_x": round(artwork["position"]["x"], 3), "pos_z": round(artwork["position"]["z"], 3), "theta": round(artwork["rotation"]["eulerAngles"]["y"]), "_theta": round(artwork["rotation"]["eulerAngles"]["y"])} for artwork in exhibition_data["paintings"]] #작품 id 리스트 #TODO

def cal_dist(x1, y1, x2, y2, a, b):
    area = abs((x1-a) * (y2-b) - (y1-b) * (x2 - a))
    AB = ((x1-x2)**2 + (y1-y2)**2) **0.5
    distance = area/AB
    return distance


for exhibited_artwork in exhibited_artwork_list:
    for artwork in artwork_list:
        if exhibited_artwork["id"] == artwork["id"]:
            exhibited_artwork["artist"] = artwork["artist"]
            exhibited_artwork["width"] = round(float(artwork["size"].split("x")[1]) / 100, 3) #width 정보만 빼오기
            # exhibited_artwork["height"] = round(float(artwork["size"].split("x")[0]) / 100, 3) #height 정보만 빼오기 #현재는 사용하지 않음.
            # exhibited_artwork["placed"] = False
            # exhibited_artwork["size"] = round(float(artwork["size"].split("x")) / 100, 3) # split 잘 먹었는지 확인용
    
    exhibited_artwork["theta"] = (round(exhibited_artwork["theta"], 0)) % 360 #TODO
    
    #2023년 광주시립미술관 전시 예외 사항
    if(exhibited_artwork['id'] == 'PA-0101'):
        exhibited_artwork['theta'] = 0
    if(exhibited_artwork['id'] == 'PA-0065'):
        exhibited_artwork['wall'] = 'w45'
    if(exhibited_artwork['id'] == 'PA-0070'):
        exhibited_artwork['wall'] = 'w45'

    for wall in wall_list: #TODO #작품에 오일러앵글 쓰자! 올해는!! 2023년 광주시립미술관은 수정 필요
        #wall["hanged_artwork"] = []
        distance = cal_dist(wall["x1"], wall["z1"], wall["x2"], wall["z2"], exhibited_artwork["pos_x"], exhibited_artwork["pos_z"])
        if (distance < 0.1) and (abs(exhibited_artwork["theta"] - wall["theta"])<5):
            if math.dist((exhibited_artwork["pos_x"], exhibited_artwork["pos_z"]), ((wall["x1"] + wall["x2"])/2, (wall["z1"] + wall["z2"])/2)) < wall["length"]/2:
                exhibited_artwork["_theta"] = wall["theta"]
                exhibited_artwork["wall"] = wall["id"]
                if exhibited_artwork['id'] not in wall["hanged_artwork"]:
                    wall["hanged_artwork"].append(exhibited_artwork['id'])

    print(exhibited_artwork)

def heatmap_generator(
    artwork_width, new_pos_x, new_pos_z, old_pos_x, old_pos_z, x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap
):
    #작품 객체 indicate heatmap 생성
    x1, x2, z1, z2 = new_pos_x - artwork_width/2, new_pos_x + artwork_width/2, new_pos_z, new_pos_z
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_heatmap = np.zeros((400, 400), dtype = np.int16)
    artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #작품 프레임 두께 10cm
    
    #작품이 향하고 있는 방향 표시
    direction = np.zeros((400, 400), dtype = np.int16)
    direction = cv2.line(direction, (_x1, _z1), (round((_x1+_x2)/2), round((_z1+_z2)/2)), 255, 1)
    direction_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 90, 0.7)
    direction = cv2.warpAffine(direction, direction_rotation, direction.shape)
    artwork_heatmap += direction

    #작품 새로운 벽 방향으로 회전
    artwork_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta, 1)
    artwork_heatmap = cv2.warpAffine(artwork_heatmap, artwork_rotation, artwork_heatmap.shape)
    #artwork_heatmap[artwork_heatmap > 0] = -10 # 확인용
    artwork_heatmap[artwork_heatmap > 0] = 0 # 분산 계산용

    #작품 관람객 히트맵 회전 #TODO
    
    #artwork_visitor_rotation = cv2.getRotationMatrix2D((round(old_pos_x/heatmap_cell_size), round(old_pos_z/heatmap_cell_size)), 0, 1)
    #artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_visitor_transform = np.float32([[1, 0, round((new_pos_x - old_pos_x)/heatmap_cell_size)], [0, 1, round((new_pos_z - old_pos_z)/heatmap_cell_size)]])
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)
    #artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta - old_theta, 1)
    #artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_heatmap += artwork_visitor_heatmap

    return artwork_heatmap


artwork_location_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16) #1px == 10cm 크기 
x_offset, z_offset = 7, 12

for exhibited_artwork in exhibited_artwork_list:
    artwork_visitor_heatmap = np.load('Daegu_new_preAURA_2023+(9-22)/'+ exhibited_artwork["id"] + '.npy')
    artwork_heatmap = heatmap_generator(exhibited_artwork["width"], exhibited_artwork["pos_x"], exhibited_artwork["pos_z"], exhibited_artwork["pos_x"], exhibited_artwork["pos_z"], x_offset, z_offset, heatmap_cell_size, 0, exhibited_artwork["_theta"], artwork_visitor_heatmap) #TODO
    artwork_location_heatmap += artwork_heatmap

for wall in wall_list:
    wall["artwork"] = []
    if(wall["id"] == "w45"):
        wall["hanged_artwork"] = ["PA-0070", "PA-0065"]
    for order in exhibited_artwork_order:
        if order in wall["hanged_artwork"]:
            if order not in wall["artwork"]:
                wall["artwork"].append(order)
    # del wall["hanged_artwork"]
    print(wall)

with open('wall_list_2023.pkl', 'wb') as f:
    pickle.dump(wall_list,f)

ordered_exhibited_artwork_list = []
for order in exhibited_artwork_order:
    for exhibited_artwork in exhibited_artwork_list:
        if order == exhibited_artwork["id"]:
            ordered_exhibited_artwork_list.append(exhibited_artwork)

with open('exhibited_artwork_list_2023.pkl', 'wb') as f:
    pickle.dump(ordered_exhibited_artwork_list,f)



heatmap = space_heatmap + artwork_location_heatmap
np.save("initial_heatmap_2023", heatmap)
heatmapCSV = pd.DataFrame(heatmap)
sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-10, vmax=50)
plt.show()











'''
#작품들 위치 히트맵 위에 표시
for exhibited_artwork in exhibited_artwork_list:
    x1, x2, z1, z2 = exhibited_artwork["pos_x"] - exhibited_artwork["width"]/2, exhibited_artwork["pos_x"] + exhibited_artwork["width"]/2, exhibited_artwork["pos_z"], exhibited_artwork["pos_z"]
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #작품 프레임 두께 10cm
    
    #작품이 향하고 있는 방향 표시
    direction = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    direction = cv2.line(direction, (_x1, _z1), (round((_x1+_x2)/2), round((_z1+_z2)/2)), 255, 1)
    direction_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 90, 0.7)
    direction = cv2.warpAffine(direction, direction_rotation, direction.shape)
    artwork_heatmap += direction

    #벽 방향으로 회전
    artwork_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), exhibited_artwork["theta"], 1)
    artwork_heatmap = cv2.warpAffine(artwork_heatmap, artwork_rotation, artwork_heatmap.shape)
    
    # _artwork_heatmap = pd.DataFrame(artwork_heatmap)
    # sns.heatmap(_artwork_heatmap, cmap='RdYlGn_r', vmin=0, vmax=510)
    # plt.show()

    if(exhibited_artwork["id"] == "PA-0024"):
        artwork_visitor_heatmap = np.load('Daegu_new_preAURA_1025_1117+(8-13)/'+ exhibited_artwork["id"] + '.npy')
        
        artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 0, 1)
        artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)
        artwork_visitor_transform = np.float32([[1, 0, 0], [0, 1, 0]])
        artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)

        artwork_heatmap = artwork_heatmap + artwork_visitor_heatmap
        
        artwork_location_heatmap += artwork_heatmap
    
artwork_location_heatmap[artwork_location_heatmap > 250] = -30 #회화 작품 객체 -20으로 표시해서 확인 용도
'''



