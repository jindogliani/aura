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

def goal_cost(optimized_artwork_heatmap):
    # 음수는 값에 안 넣게 필터 필요
    variance = np.var(optimized_artwork_heatmap[optimized_artwork_heatmap>=0])
    return variance

def regularization_cost(exhibited_artwork_list): #생성된 씬의 작품들의 coords
    new_artwork_positions = []
    new_artwork_distance = []
    for wall in exhibited_artwork_list:
        new_artwork_positions.append((wall['pos_x'], wall['pos_z'])) 
    for i in range(len(new_artwork_positions)):
        j = (i+1) % len(new_artwork_positions)
        new_artwork_distance.append(math.dist(new_artwork_positions[i], new_artwork_positions[j]))
    new_artwork_distance_arr = np.array(new_artwork_distance)
    variance = np.var(new_artwork_distance_arr)
    return variance #노멀라이즈 이전

def centroid(coordinates):
    x_coords = [p[0] for p in coordinates]
    y_coords = [p[1] for p in coordinates]
    _len = len(coordinates)
    if _len == 0:
        return (0, 0)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return (centroid_x, centroid_y)

def similarity_cost(optimized_artwork_list):
    artist_list = ["차현욱", "백지훈", "이주희", "신준민", "박지연", "황지영", "이연주", "심윤", "김도경", "하지"] #작가 수가 10 미만이라 그냥 리스트 만들고 진행, 나중에 돌리기 전에 지정
    cluster_variance_list = []
    for artist in artist_list:
        same_artist_coords_list = []
        same_artist_dist_list = []

        for artwork in optimized_artwork_list:
            if artwork['artist'] == artist:
                same_artist_coords_list.append((artwork['pos_x'], artwork['pos_z']))
        if same_artist_coords_list == []:
             same_artist_coords_list.append((0,0))
        
        centroid_coords = centroid(same_artist_coords_list)

        for coords in same_artist_coords_list:
            same_artist_dist_list.append(math.dist(coords, centroid_coords))

        same_artist_dist_arr = np.array(same_artist_dist_list)
        same_artist_dist_var = np.var(same_artist_dist_arr)
        cluster_variance_list.append(same_artist_dist_var)

    WCSS = sum(cluster_variance_list)

    return WCSS

ver = "2022"

init_cell_variance = 2000
init_regulation_variance = 2000
init_WCSS = 2000

if ver == "2023":
    with open('2023_wall_list_with_artworks.pkl', 'rb') as f:
        wall_list = pickle.load(f)
    with open('2023_exhibited_artwork_list.pkl', 'rb') as f:
        exhibited_artwork_list = pickle.load(f)
    initial_heatmap = np.load("2023_initial_heatmap.npy")

    init_cell_variance = goal_cost(initial_heatmap)
    init_regulation_variance = regularization_cost(exhibited_artwork_list)
    init_WCSS = similarity_cost(exhibited_artwork_list)

    print("2023 Initial Density per Cell Variance: " + str(init_cell_variance))
    print("2023 Initial Artwork Distance Variance: " + str(init_regulation_variance))
    print("2023 Initial WCSS: " + str(init_WCSS))

elif ver == "2022":
    with open('2022_wall_list_with_artworks.pkl', 'rb') as f:
        wall_list = pickle.load(f)
    with open('2022_exhibited_artwork_list.pkl', 'rb') as f:
        exhibited_artwork_list = pickle.load(f)
    initial_heatmap = np.load("2022_initial_heatmap.npy")
    
    init_cell_variance = goal_cost(initial_heatmap)
    init_regulation_variance = regularization_cost(exhibited_artwork_list)
    init_WCSS = similarity_cost(exhibited_artwork_list)

    print("2022 Initial Density per Cell Variance: " + str(init_cell_variance))
    print("2022 Initial Artwork Distance Variance: " + str(init_regulation_variance))
    print("2022 Initial WCSS: " + str(init_WCSS))