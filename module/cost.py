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

with open('../wall_list_2023.pkl', 'rb') as f:
    wall_list = pickle.load(f)

with open('../exhibited_artwork_list_2023.pkl', 'rb') as f:
    exhibited_artwork_list = pickle.load(f)

space_heatmap = np.load('../SpaceData/coords_GMA3+(9-20).npy')
space_heatmap[space_heatmap > 254] = -55 #공간 벽을 -10으로 변환
space_heatmap[space_heatmap == 0] = -50 #공간 외부 값을 0에서 -15으로 전환
space_heatmap[space_heatmap == 127] = 0 #공간 내부 값을 127에서 0으로 전환

init_cell_variance = 28.96355947950509
init_regulation_variance = 5.866548641795766
init_WCSS = 52.517175423485185

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
    #artwork_heatmap[artwork_heatmap > 0] = -6 # 확인용
    artwork_heatmap[artwork_heatmap > 0] = 0 # 분산 계산용

    #작품 관람객 히트맵 회전 #TODO
    
    #artwork_visitor_rotation = cv2.getRotationMatrix2D((round(old_pos_x/heatmap_cell_size), round(old_pos_z/heatmap_cell_size)), 0, 1)
    #artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_visitor_transform = np.float32([[1, 0, round((new_pos_x - old_pos_x)/heatmap_cell_size)], [0, 1, round((new_pos_z - old_pos_z)/heatmap_cell_size)]])
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)
    artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta - old_theta, 1)
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_heatmap += artwork_visitor_heatmap

    return artwork_heatmap

def goal_cost(scene_data, artwork_data, wall_data):
    # 음수는 값에 안 넣게 필터 필요
    scene_heatmap = np.zeros((400, 400), dtype = np.int16)
    x_offset, z_offset = 7, 12
    
    for k, v in scene_data.items():
        art = artwork_data[k]
        wall = wall_data[v[0]]
        pos = v[1]
        
        ratio1, ratio2 = (wall["length"]-pos)/wall["length"], pos/wall["length"]
        new_art_pos = (wall["x1"]*ratio1+wall["x2"]*ratio2, wall["z1"]*ratio1+wall["z2"]*ratio2)
        old_art_pos = (art["pos_x"], art["pos_z"])

        new_theta = wall["theta"]
        old_theta = art["theta"]

        heatmap_cell_size = 0.1

        artwork_visitor_heatmap = np.load('Daegu_new_preAURA_2023+(9-20)/'+ k + '.npy')
        artwork_heatmap = heatmap_generator(art["width"], new_art_pos[0], new_art_pos[1], old_art_pos[0], old_art_pos[1], x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap)
        scene_heatmap += artwork_heatmap
    
    scene_heatmap += space_heatmap
    variance = np.var(scene_heatmap[scene_heatmap>=0])
    cost = variance / init_cell_variance
    
    return cost

def regularization_cost(scene_data, artwork_data, wall_data): #생성된 씬의 작품들의 coords
    ordered_scene_data = []
    
    for wall in wall_data:
        for k, v in scene_data.items():
            art = artwork_data[k]
            hanged_wall = wall_data[v[0]]
            pos = v[1]
            
            ratio1, ratio2 = (hanged_wall["length"]-pos)/hanged_wall["length"], pos/hanged_wall["length"]
            new_art_pos = (hanged_wall["x1"]*ratio1+hanged_wall["x2"]*ratio2, hanged_wall["z1"]*ratio1+hanged_wall["z2"]*ratio2)
            
            if wall["id"] == hanged_wall["id"]:
                dic = [art["id"], wall["id"], pos, new_art_pos]
                if ordered_scene_data == []:
                    ordered_scene_data.append(dic)
                else:
                    for i, e in enumerate(reversed(ordered_scene_data)):
                        if e[1] == dic[1]:
                            if e[2] <= dic[2]:
                                if i > 0:
                                    # reversed(ordered_scene_data).insert(dic, i-1)
                                    ordered_scene_data.insert(dic, -i) # 뭐가 맞는지 모르겠음..
                                else:
                                    ordered_scene_data.append(dic)
                        else:
                            ordered_scene_data.append(dic)
                        
    new_artwork_positions = []
    new_artwork_distance = []
    
    for art in ordered_scene_data:
        new_artwork_positions.append(art[3]) 
    
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
            
            ratio1, ratio2 = (wall["length"]-pos)/wall["length"], pos/wall["length"]
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
