"""
- 3 cost functions
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

# goal cost function 에 들어감
optimized_artwork_heatmap = np.load('optimized_artwork_heatmap.npy')

# regularization function 에 들어감
with open('optimized_wall_list_sample.pkl', 'rb') as f:
    optimized_wall_list = pickle.load(f)

# artwork similarity function 에 들어감
with open('optimized_artwork_list_sample.pkl', 'rb') as f:
    optimized_artwork_list = pickle.load(f)

# 작가 정보를 아트워크 리스트에 안 넣어서 일단 넣음
for artwork in optimized_artwork_list:
    artwork['artist'] = 'not_set'
    #print(artwork)
for wall in optimized_wall_list:
    # print(wall)
    pass

init_cell_variance = 2000
init_regulation_variance = 2000
init_WCSS = 2000

def goal_cost(optimized_artwork_heatmap):
    # 음수는 값에 안 넣게 필터 필요
    variance = np.var(optimized_artwork_heatmap[optimized_artwork_heatmap>=0])
    return variance

def regularization_cost(optimized_wall_list):
    new_artwork_positions = []
    new_artwork_distance = []
    for wall in optimized_wall_list:
        new_artwork_positions.append(wall['hanged_artworks']['new_coords']) 
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
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return (centroid_x, centroid_y)

def similarity_cost(optimized_artwork_list):
    artist_list = ["한놈", "두식이", "석삼", "너구리", "오징어", "육개장"] #작가 수가 10 미만이라 그냥 리스트 만들고 진행, 나중에 돌리기 전에 지정
    cluster_variance_list = []
    for artist in artist_list:
        same_artist_coords_list = []
        same_artist_dist_list = []

        for artwork in optimized_artwork_list:
            if artwork['artist'] == artist:
                same_artist_coords_list.append(artwork['new_coords'])

        centroid_coords = centroid(same_artist_coords_list)

        for coords in same_artist_coords_list:
            same_artist_dist_list.append(math.dist(coords, centroid_coords))

        same_artist_dist_arr = np.array(same_artist_dist_list)
        same_artist_dist_var = np.var(same_artist_dist_arr, ddof = 1)
        cluster_variance_list.append(same_artist_dist_var)

    WCSS = sum(cluster_variance_list)

    return WCSS

# WCSS = similarity_cost(optimized_artwork_list)
# norm_WCSS = WCSS / init_WCSS
# if (norm_WCSS >= 1):
#     norm_WCSS = 1

#개삽질

arr4 = np.array([[1,2,0],[1,-1,0],[-1,-1,0]])

def _goal_cost(optimized_artwork_heatmap):
    # 음수는 값에 안 넣게 필터 필요
    
    leng = sum(sum(optimized_artwork_heatmap >= 0))
    optimized_artwork_heatmap_copy = np.where(optimized_artwork_heatmap >=0, optimized_artwork_heatmap, 0)

    mean = sum(sum(optimized_artwork_heatmap_copy)) / leng

    print(leng)
    print(mean)
    variance0 = sum(sum((optimized_artwork_heatmap - mean)**2))/leng
    variance0 = np.var(optimized_artwork_heatmap[optimized_artwork_heatmap >=0])
    print(optimized_artwork_heatmap[optimized_artwork_heatmap >=0])

    
    variance1 = np.var(optimized_artwork_heatmap[optimized_artwork_heatmap>=0])
    
    
    variance2 = np.var(optimized_artwork_heatmap[optimized_artwork_heatmap>=0])
    
    print(variance0)
    print(variance1)
    print(variance2)

_goal_cost(arr4)