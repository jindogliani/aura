"""
cost functions
"""
import os
from time import localtime, time
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import time

space_vertical_size, space_horizontal_size = 40, 40
heatmap_cell_size = 0.2
space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
space_horizontal_cells, space_vertcal_cells = round(space_horizontal_cells), round(space_vertcal_cells)

with open('wall_list_2023.pkl', 'rb') as f:
    wall_list = pickle.load(f)

with open('exhibited_artwork_list_2023.pkl', 'rb') as f:
    exhibited_artwork_list = pickle.load(f)

space_heatmap = np.load('SpaceData/coords_GMA3+(9-23).npy')
space_heatmap[space_heatmap > 254] = -1000 #공간 벽을 -10으로 변환
space_heatmap[space_heatmap == 0] = -1000 #공간 외부 값을 0에서 -15으로 전환
space_heatmap[space_heatmap == 127] = 0 #공간 내부 값을 127에서 0으로 전환

init_scene_data = {'PA-0023': ['w5', 28], 'PA-0026': ['w5', 65], 'KO-0009': ['w6', 29], 'PA-0095': ['w8', 12], 'PA-0098': ['w8', 33], 'PA-0076': ['w8', 49], 'PA-0074': ['w9', 16], 'PA-0075': ['w9', 31], 'PA-0077': ['w9', 49], 'KO-0010': ['w9', 80], 'KO-0008': ('w9', 116), 'PA-0101': ['w9', 132], 'PA-0057': ['w10', 11], 'PA-0052': ['w10', 29], 'PA-0061': ['w10', 49], 'PA-0001': ['w14', 33], 'PA-0003': ['w18', 41], 'PA-0004': ['w18', 93], 'PA-0082': ['w24', 17], 'PA-0084': ['w24', 48], 'PA-0083': ['w26', 21], 'PA-0063': ['w26', 46], 'PA-0067': ['w27', 22], 'PA-0064': ['w27', 49], 'PA-0024': ['w31', 53], 'PA-0087': ['w41', 28], 'PA-0027': ['w42', 16], 'PA-0025': ['w42', 46], 'PA-0036': ['w43', 19], 'PA-0085': ['w43', 40], 'PA-0086': ['w43', 55], 'PA-0070': ['w45', 12], 'PA-0065': ['w45', 50], 'PA-0031': ['w46', 35], 'PA-0088': ['w50', 34], 'PA-0100': ['w52', 29], 'PA-0099': ['w52', 58], 'KO-0007': ['w56', 33], 'KO-0006': ['w57', 49], 'KO-0004': ['w57', 86], 'KO-0005': ['w57', 112], 'PA-0090': ['w57', 154], 'PA-0089': ['w58', 34]}
init_wall_data = wall_list
init_artwork_data = exhibited_artwork_list

init_cell_variance = 130.7980201483807
init_regulation_variance = 5.866548641795766
init_WCSS = 52.517175423485185

def heatmap_generator(
    artwork_width, new_pos_x, new_pos_z, old_pos_x, old_pos_z, x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap
):
    #작품 객체 indicate heatmap 생성
    x1, x2, z1, z2 = new_pos_x - artwork_width/2, new_pos_x + artwork_width/2, new_pos_z, new_pos_z
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #작품 프레임 두께 10cm
    
    #작품이 향하고 있는 방향 표시
    # direction = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    # direction = cv2.line(direction, (_x1, _z1), (round((_x1+_x2)/2), round((_z1+_z2)/2)), 255, 1)
    # direction_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 90, 0.7)
    # direction = cv2.warpAffine(direction, direction_rotation, direction.shape)
    # artwork_heatmap += direction

    #작품 새로운 벽 방향으로 회전
    artwork_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta, 1)
    artwork_heatmap = cv2.warpAffine(artwork_heatmap, artwork_rotation, artwork_heatmap.shape)
    #artwork_heatmap[artwork_heatmap > 0] = -6 # 확인용
    artwork_heatmap[artwork_heatmap > 0] = 0 # 분산 계산용

    #작품 관람객 히트맵 회전 #TODO
    
    #artwork_visitor_rotation = cv2.getRotationMatrix2D((round(old_pos_x/heatmap_cell_size), round(old_pos_z/heatmap_cell_size)), 0, 1)
    #artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    # if ((new_pos_x - old_pos_x)/heatmap_cell_size > 2) or ((new_pos_x - old_pos_x)/heatmap_cell_size > 2) or abs(new_theta - old_theta) > 10:
    artwork_visitor_transform = np.float32([[1, 0, round((new_pos_x - old_pos_x)/heatmap_cell_size)], [0, 1, round((new_pos_z - old_pos_z)/heatmap_cell_size)]])
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)
    artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta - old_theta, 1)
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_heatmap += artwork_visitor_heatmap
    return artwork_heatmap

def goal_cost(scene_data, artwork_data, wall_data):
    # 음수는 값에 안 넣게 필터 필요
    scene_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    x_offset, z_offset = 7, 12

    for k, v in scene_data.items():
        art = artwork_data[k]
        wall = wall_data[v[0]]
        pos = v[1]
        
        temp_heatmap_cell_size = 0.1
        ratio1, ratio2 = (wall["length"]-pos*temp_heatmap_cell_size)/wall["length"], (pos*temp_heatmap_cell_size)/wall["length"]
        new_art_pos = (wall["x1"]*ratio1+wall["x2"]*ratio2, wall["z1"]*ratio1+wall["z2"]*ratio2)
        old_art_pos = (art["pos_x"], art["pos_z"])

        new_theta = wall["theta"]
        old_theta = art["theta"]
        artwork_visitor_heatmap = np.load('Daegu_new_preAURA_2023+(9-23)/'+ k + '.npy')
        artwork_heatmap = heatmap_generator(art["width"], new_art_pos[0], new_art_pos[1], old_art_pos[0], old_art_pos[1], x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap)
        scene_heatmap += artwork_heatmap
    
    scene_heatmap += space_heatmap

    # heatmapCSV = pd.DataFrame(scene_heatmap)
    # sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-10, vmax=50)
    # plt.show()

    variance = np.var(scene_heatmap[scene_heatmap>=0])
    cost = variance / init_cell_variance
    # cost = init_cell_variance - variance
    
    return cost

def regularization_cost(scene_data, artwork_data, wall_data): #생성된 씬의 작품들의 coords
    keys, values = list(scene_data.keys()), list(scene_data.values())
    ordered_scene_data = {k:v for v, k in sorted(zip(values, keys), key=(lambda x : (int(x[0][0][1:]), x[0][1])))}

    new_artwork_positions = []
    new_artwork_distance = []

    for k, v in ordered_scene_data.items():
        art = artwork_data[k]
        wall = wall_data[v[0]]
        pos = v[1]

        temp_heatmap_cell_size = 0.1
        ratio1, ratio2 = (wall["length"]-pos*temp_heatmap_cell_size)/wall["length"], (pos*temp_heatmap_cell_size)/wall["length"]
        new_art_pos = (wall["x1"]*ratio1+wall["x2"]*ratio2, wall["z1"]*ratio1+wall["z2"]*ratio2)
        new_artwork_positions.append(new_art_pos)
    
    for i in range(len(new_artwork_positions)):
        j = (i+1) % len(new_artwork_positions)
        new_artwork_distance.append(math.dist(new_artwork_positions[i], new_artwork_positions[j]))
    
    new_artwork_distance_arr = np.array(new_artwork_distance)
    variance = np.var(new_artwork_distance_arr)
    cost = variance / init_regulation_variance
    return cost

def centroid(coordinates):
    x_coords = [p[0] for p in coordinates]
    y_coords = [p[1] for p in coordinates]
    _len = len(coordinates)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return (centroid_x, centroid_y)

def similarity_cost(scene_data, artwork_data, wall_data):
    artist_list = ["차현욱", "백지훈", "이주희", "신준민", "박지연", "황지영", "이연주", "심윤", "김도경"] #작가 수가 10 미만이라 그냥 리스트 만들고 진행, 나중에 돌리기 전에 지정
    cluster_variance_list = []
    
    for artist in artist_list:
        same_artist_coords_list = []
        same_artist_dist_list = []

        for k, v in scene_data.items():
            art = artwork_data[k]
            wall = wall_data[v[0]]
            pos = v[1]
            
            temp_heatmap_cell_size = 0.1
            ratio1, ratio2 = (wall["length"]-pos*temp_heatmap_cell_size)/wall["length"], (pos*temp_heatmap_cell_size)/wall["length"]
            new_art_pos = (wall["x1"]*ratio1+wall["x2"]*ratio2, wall["z1"]*ratio1+wall["z2"]*ratio2)

            if art["artist"] == artist:
                same_artist_coords_list.append(new_art_pos)

        centroid_coords = centroid(same_artist_coords_list)

        for coords in same_artist_coords_list:
            same_artist_dist_list.append(math.dist(coords, centroid_coords))

        same_artist_dist_arr = np.array(same_artist_dist_list)
        same_artist_dist_var = np.var(same_artist_dist_arr)
        cluster_variance_list.append(same_artist_dist_var)

    WCSS = sum(cluster_variance_list)
    cost = WCSS / init_WCSS

    return cost

