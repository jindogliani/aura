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

with open('wall_list_2023.pkl', 'rb') as f:
    wall_list = pickle.load(f)

with open('exhibited_artwork_list_2023.pkl', 'rb') as f:
    exhibited_artwork_list = pickle.load(f)

initial_heatmap = np.load("initial_heatmap_2023.npy")

init_cell_variance = 2000
init_regulation_variance = 2000
init_WCSS = 2000

exhibited_artwork_order = ["PA-0023", "PA-0026", "KO-0009", "PA-0095", "PA-0098", "PA-0076", "PA-0074", "PA-0075", "PA-0077", "KO-0010", "KO-0008", "PA-0101", "PA-0057", "PA-0052", "PA-0061", "PA-0001", "PA-0003", "PA-0004", "PA-0082", "PA-0084", "PA-0083", "PA-0063", "PA-0067", "PA-0064", "PA-0024", "PA-0087", "PA-0027", "PA-0025", "PA-0036", "PA-0085", "PA-0086", "PA-0070", "PA-0065", "PA-0031", "PA-0088", "PA-0100", "PA-0099", "KO-0007", "KO-0006", "KO-0004", "KO-0005", "PA-0090", "PA-0089"]
ordered_exhibited_artwork_list = []

for order in exhibited_artwork_order:
    for exhibited_artwork in exhibited_artwork_list:
        if order == exhibited_artwork["id"]:
            ordered_exhibited_artwork_list.append(exhibited_artwork)

for a in ordered_exhibited_artwork_list:
    print(a)

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
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return (centroid_x, centroid_y)

def similarity_cost(optimized_artwork_list):
    artist_list = ["차현욱", "백지훈", "이주희", "신준민", "박지연", "황지영", "이연주", "심윤", "김도경"] #작가 수가 10 미만이라 그냥 리스트 만들고 진행, 나중에 돌리기 전에 지정
    cluster_variance_list = []
    for artist in artist_list:
        same_artist_coords_list = []
        same_artist_dist_list = []

        for artwork in optimized_artwork_list:
            if artwork['artist'] == artist:
                same_artist_coords_list.append((artwork['pos_x'], artwork['pos_z']))

        centroid_coords = centroid(same_artist_coords_list)

        for coords in same_artist_coords_list:
            same_artist_dist_list.append(math.dist(coords, centroid_coords))

        same_artist_dist_arr = np.array(same_artist_dist_list)
        same_artist_dist_var = np.var(same_artist_dist_arr)
        cluster_variance_list.append(same_artist_dist_var)

    WCSS = sum(cluster_variance_list)

    return WCSS

init_cell_variance = goal_cost(initial_heatmap)
_init_regulation_variance = regularization_cost(exhibited_artwork_list)
init_regulation_variance = regularization_cost(ordered_exhibited_artwork_list)
init_WCSS = similarity_cost(exhibited_artwork_list)

print("Initial Density per Cell Variance: " + str(init_cell_variance))
print("Initial Artwork Distance Variance: " + str(init_regulation_variance))
print("Initial WCSS: " + str(init_WCSS))

with open('exhibited_artwork_list_2023.pkl', 'wb') as f:
    pickle.dump(ordered_exhibited_artwork_list,f)


'''
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
'''
