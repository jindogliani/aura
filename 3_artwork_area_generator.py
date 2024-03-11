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

date = '+' + '(' + str(localtime(time()).tm_mon) +'-'+ str(localtime(time()).tm_mday) +'-'+ str(localtime(time()).tm_hour) + '-'+ str(localtime(time()).tm_min) + ')'

def cal_dist(x1, y1, x2, y2, a, b):
    area = abs((x1-a) * (y2-b) - (y1-b) * (x2 - a))
    AB = ((x1-x2)**2 + (y1-y2)**2) **0.5
    distance = area/AB
    return distance


def heatmap_generator(
    visualize_mode, artwork_width, new_pos_x, new_pos_z, old_pos_x, old_pos_z, x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap, space_horizontal_cells, space_vertcal_cells
):
    #작품 객체 indicate heatmap 생성
    x1, x2, z1, z2 = new_pos_x - artwork_width/2, new_pos_x + artwork_width/2, new_pos_z, new_pos_z
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    if new_theta in (0, 90, 180, 270):
        artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #작품 프레임 두께 20cm
    else:
        artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #작품 프레임 두께 10cm

    #작품이 향하고 있는 방향 표시
    direction = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    direction = cv2.line(direction, (_x1, _z1), (round((_x1+_x2)/2), round((_z1+_z2)/2)), 255, 1)
    direction_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 90, 0.7)
    direction = cv2.warpAffine(direction, direction_rotation, direction.shape)
    # artwork_heatmap += direction # 작품 방향 이제 표시 X 02.26

    #작품 새로운 벽 방향으로 회전
    artwork_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta, 1)
    artwork_heatmap = cv2.warpAffine(artwork_heatmap, artwork_rotation, artwork_heatmap.shape)
    if(visualize_mode):
        artwork_intensity = 15
        artwork_heatmap[artwork_heatmap > 0] = artwork_intensity # 확인용
    else:
        artwork_heatmap[artwork_heatmap > 0] = 0 # 분산 계산용

    artwork_visitor_transform = np.float32([[1, 0, round((new_pos_x - old_pos_x)/heatmap_cell_size)], [0, 1, round((new_pos_z - old_pos_z)/heatmap_cell_size)]])
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)
    artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta - old_theta, 1)
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_heatmap += artwork_visitor_heatmap
    return artwork_heatmap


def space_artwork_visitor_merge(ver, visualize_mode, wall_list, space_heatmap, total_artwork_list, exhibition_data_path, exhibited_artwork_order, x_offset, z_offset, space_vertical_size, space_horizontal_size, heatmap_cell_size):
    
    space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
    space_horizontal_cells, space_vertcal_cells = round(space_horizontal_cells), round(space_vertcal_cells)

    if(visualize_mode == False):
        space_heatmap[space_heatmap > 254] = -1000 #공간 벽을 -1000으로 변환
        space_heatmap[space_heatmap == 0] = -1000 #공간 외부 값을 0에서 -1000으로 전환
        space_heatmap[space_heatmap == 127] = 0 #공간 내부 값을 127에서 0으로 전환
    else:
        space_heatmap[space_heatmap > 254] = gallery_wall_intensity #공간 벽을 -6으로 변환
        space_heatmap[space_heatmap == 0] = gallery_outside_intensity #공간 외부 값을 0에서 -3으로 전환
        space_heatmap[space_heatmap == 127] = gallery_inside_intensity #공간 내부 값을 127에서 0으로 전환

    exhibited_artwork_list =[]
    with open(exhibition_data_path, 'r', -1, encoding='utf-8') as f:
        exhibition_data = json.load(f)

    exhibited_artwork_list = [{"id": artwork["artworkIndex"], "wall": "-", "pos_x": round(artwork["position"]["x"], 3), "pos_z": round(artwork["position"]["z"], 3), "theta": round(artwork["rotation"]["eulerAngles"]["y"]), "_theta": round(artwork["rotation"]["eulerAngles"]["y"])} for artwork in exhibition_data["paintings"]]

    for exhibited_artwork in exhibited_artwork_list:
        exhibited_artwork["theta"] = (round(exhibited_artwork["theta"], 0)) % 360
        if abs(exhibited_artwork['theta'] - 360) <= 5:
            exhibited_artwork['theta'] = 0
        for artwork in total_artwork_list:
            if exhibited_artwork["id"] == artwork["id"]:
                exhibited_artwork["artist"] = artwork["artist"]
                exhibited_artwork["width"] = round(float(artwork["size"].split("x")[1]) / 100, 3) #width 정보만 빼오기
                # exhibited_artwork["height"] = round(float(artwork["size"].split("x")[0]) / 100, 3) #height 정보만 빼오기 #현재는 사용하지 않음.
                # exhibited_artwork["placed"] = False
                # exhibited_artwork["size"] = round(float(artwork["size"].split("x")) / 100, 3) # split 잘 먹었는지 확인용
        
        if(ver == "2023"):
            #2023년 광주시립미술관 전시 예외 사항
            if(exhibited_artwork['id'] == 'PA-0101'):
                exhibited_artwork['theta'] = 0
            if(exhibited_artwork['id'] == 'PA-0065'):
                exhibited_artwork['wall'] = 'w45'
            if(exhibited_artwork['id'] == 'PA-0070'):
                exhibited_artwork['wall'] = 'w45'
            if(exhibited_artwork['id'] == 'PA-0087'):
                exhibited_artwork['wall'] = 'w41'

        for wall in wall_list:
            if "hanged_artwork" not in wall:
                wall["hanged_artwork"] = []
            distance = cal_dist(wall["x1"], wall["z1"], wall["x2"], wall["z2"], exhibited_artwork["pos_x"], exhibited_artwork["pos_z"])
            if (distance < 0.2) and (abs(exhibited_artwork["theta"] - wall["theta"])<5):
                if math.dist((exhibited_artwork["pos_x"], exhibited_artwork["pos_z"]), ((wall["x1"] + wall["x2"])/2, (wall["z1"] + wall["z2"])/2)) < wall["length"]/2:
                    exhibited_artwork["_theta"] = wall["theta"]
                    exhibited_artwork["wall"] = wall["id"]
                    if exhibited_artwork['id'] not in wall["hanged_artwork"]:
                        wall["hanged_artwork"].append(exhibited_artwork['id'])

    artwork_location_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16) #1px == 10cm 크기 

    for exhibited_artwork in exhibited_artwork_list:
        if ver == "2023":
            artwork_visitor_heatmap = np.load('Data_2023_preAURA_2023+(9-24-17-25)/'+ exhibited_artwork["id"] + '.npy') #cell size 0.1
        elif ver == "2022":
            artwork_visitor_heatmap = np.load('Data_2022_preAURA_2022+(9-27-19-59)/'+ exhibited_artwork["id"] + '.npy') #cell size 0.1
        
        artwork_heatmap = heatmap_generator(visualize_mode, exhibited_artwork["width"], exhibited_artwork["pos_x"], exhibited_artwork["pos_z"], exhibited_artwork["pos_x"], exhibited_artwork["pos_z"], x_offset, z_offset, heatmap_cell_size, exhibited_artwork["_theta"], exhibited_artwork["_theta"], artwork_visitor_heatmap, space_horizontal_cells, space_vertcal_cells) #TODO
        artwork_location_heatmap += artwork_heatmap

    for wall in wall_list:
        wall["length"] = round(wall["length"], 3)
        wall["artwork"] = []
        if(ver == "2023"):
            wall["x1"], wall["z1"], wall["x2"], wall["z2"] = round(wall["x1"], 3), round(wall["z1"], 3), round(wall["x2"], 3), round(wall["z2"], 3)
            no_display_walls = ["w1", "w21", "w36", "w38"]
            if wall["id"] in no_display_walls:
                wall["displayable"] = False
            if(wall["id"] == "w41"):
                wall["hanged_artwork"] = ["PA-0087"]
            if(wall["id"] == "w45"):
                wall["hanged_artwork"] = ["PA-0070", "PA-0065"]
        if(ver == "2022"):
            no_display_walls = ["w10", "w11", "w12", "w13", "w21", "w22", "w35", "w36"]
            if wall["id"] in no_display_walls:
                wall["displayable"] = False
        for order in exhibited_artwork_order:
            if order in wall["hanged_artwork"]:
                if order not in wall["artwork"]:
                    wall["artwork"].append(order)
        # del wall["hanged_artwork"]
        print(wall)

    ordered_exhibited_artwork_list = []
    for order in exhibited_artwork_order:
        for exhibited_artwork in exhibited_artwork_list:
            if order == exhibited_artwork["id"]:
                ordered_exhibited_artwork_list.append(exhibited_artwork)

    for a in ordered_exhibited_artwork_list:
        print(a)
    print(len(ordered_exhibited_artwork_list))

    heatmap = space_heatmap + artwork_location_heatmap

    _ver = ""
    if (visualize_mode):
        _ver = ver + date + '_vis'
        if(ver == "2022"):
            rows, cols = heatmap.shape
            new_arr = np.full_like(heatmap, gallery_outside_intensity)
            shift = 65
            new_arr[shift:rows, :] = heatmap[0:rows-shift, :]
            heatmap = new_arr
    else:
        _ver = ver + date

    # np.save(_ver + "_initial_heatmap", heatmap)

    if LCLmode:
        LCL_illustrator(ver, heatmap)
    else:
        heatmap = np.flipud(heatmap)
        heatmapCSV = pd.DataFrame(heatmap)

        plt.figure(figsize=(10, 8), dpi=100)
        sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=gallery_wall_intensity, vmax=30) # RdYlGn_r or rainbow
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('visualize/' + ver + date + '_pre_UIpC_heatmap.png')

    plt.show()

    # with open(_ver + '_exhibited_artwork_list.pkl', 'wb') as f:
    #     pickle.dump(ordered_exhibited_artwork_list,f)

    # with open(_ver + '_wall_list_with_artworks.pkl', 'wb') as f:
    #     pickle.dump(wall_list,f)

def LCL_illustrator(ver, heatmap):
    
    if(ver == "2023"): #왜 2023은 혼자 y축이 반전되었을까?
            rows, cols = heatmap.shape
            new_arr = np.full_like(heatmap, gallery_outside_intensity)
            shift = 5
            new_arr[shift:rows, :] = heatmap[0:rows-shift, :]
            heatmap = new_arr

    # x, y 좌표 그리드 생성
    heatmap = np.where(heatmap < 0, 0, heatmap)
    x = np.arange(heatmap.shape[1])
    y = np.arange(heatmap.shape[0])
    x, y = np.meshgrid(x, y)

    # (x, y) 좌표와 해당 intensity 값을 포함하는 DataFrame 생성
    df = pd.DataFrame({
        'x': x.ravel(),
        'y': y.ravel(),
        'intensity': heatmap.ravel()
    })

    plt.figure(figsize=(10, 10), dpi=300)

    img_bgr = cv2.imread("SpaceData/" + ver + "_gallery_2k.png", cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb,  extent=[0, 400, 0, 400])
    
    # 등고선 KDE 플롯 생성
    sns.kdeplot(data=df, x='x', y='y', weights='intensity',
                fill=True, bw_adjust=0.3, levels=120,
                cmap="RdYlGn_r", alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.gca().invert_yaxis()  # y축 방향을 뒤집어 배열 인덱스와 일치시킴
    plt.savefig('visualize/' + ver + date + '_pre_LCL.png', transparent=True)

ver = "2023"
visualize_mode = True
LCLmode = False

space_vertical_size, space_horizontal_size = 40, 40
heatmap_cell_size = 0.1

gallery_outside_intensity = -2
gallery_wall_intensity = -4
gallery_inside_intensity = 0

artwork_data_path = "Daegu_new.json"
with open(artwork_data_path, "r", encoding='UTF8') as f:
    artwork_data = json.load(f)
total_artwork_list = [{"id": artwork["id"], "size": artwork["dimensions"], "artist": artwork["artists"][0]} for artwork in artwork_data["exhibitionObjects"]]

if ver == "2023":
    with open('2023_wall_list.pkl', 'rb') as f:
        wall_list = pickle.load(f)
    
    space_heatmap = np.load('SpaceData/2023_GMA+(9-27-14-50).npy')
    exhibition_data_path = "Data_2023.json"

    exhibited_artwork_order = ["PA-0023", "PA-0026", "PA-0095", "PA-0098", "PA-0074", "PA-0075", "PA-0077", "KO-0010", "KO-0008", "PA-0101", "PA-0052", "PA-0061", "PA-0001", "PA-0003", "PA-0004", "PA-0082", "PA-0084", "PA-0083", "PA-0063", "PA-0067", "PA-0064", "PA-0024", "PA-0087", "PA-0027", "PA-0025", "PA-0036", "PA-0085", "PA-0086", "PA-0070", "PA-0065", "PA-0031", "PA-0088", "PA-0100", "PA-0099", "KO-0007", "KO-0006", "KO-0004", "KO-0005", "PA-0090", "PA-0089"]
    print(len(exhibited_artwork_order))

    x_offset, z_offset = 7, 12 

    space_artwork_visitor_merge(ver, visualize_mode, wall_list, space_heatmap, total_artwork_list, exhibition_data_path, exhibited_artwork_order, x_offset, z_offset, space_vertical_size, space_vertical_size, heatmap_cell_size)

elif ver == "2022":
    with open('2022_wall_list.pkl', 'rb') as f:
        wall_list = pickle.load(f)
    
    space_heatmap = np.load('SpaceData/2022_HJW+(9-27-20-52).npy')
    exhibition_data_path = "Data_2022.json"

    exhibited_artwork_order = ["PA-0064", "PA-0067", "PA-0027", "PA-0025", "PA-0087", "PA-0070", "PA-0086", "PA-0024", "PA-0085", "PA-0063", "PA-0083", "PA-0072", "PA-0081", "PA-0075", "PA-0074", "PA-0052", "PA-0061", "PA-0098", "PA-0095", "PA-0093", "PA-0092", "PA-0026", "KO-0004", "KO-0001", "PA-0084", "PA-0088", "PA-0090", "PA-0089", "PA-0019", "PA-0017", "PA-0020", "PA-0056", "PA-0060", "PA-0013", "PA-0039"]
    print(len(exhibited_artwork_order))

    x_offset, z_offset = 25, 20

    space_artwork_visitor_merge(ver, visualize_mode, wall_list, space_heatmap, total_artwork_list, exhibition_data_path, exhibited_artwork_order, x_offset, z_offset, space_vertical_size, space_vertical_size, heatmap_cell_size)
