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

with open('wall_list_2022.pkl', 'rb') as f:
    wall_list = pickle.load(f)

with open('optimized_wall_list_sample.pkl', 'rb') as f:
    optimized_wall_list = pickle.load(f)

with open('optimized_artwork_list_sample.pkl', 'rb') as f:
    optimized_artwork_list = pickle.load(f)

sample_scene_dict = dict()

optimized_artwork_heatmap = np.load('optimized_artwork_heatmap.npy')

for artwork in optimized_artwork_list:
    artwork['artist'] = 'not_set'
    sample_scene_dict[artwork['id']] = (artwork['wall'], random.randint(2, 50))
    print(artwork)

for wall in optimized_wall_list:
    print(wall)

heatmapCSV = pd.DataFrame(optimized_artwork_heatmap)
sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-20, vmax=100)
plt.show()

def goal_cost(optimized_artwork_heatmap):
    # 음수는 값에 안 넣게 필터 필요
    variance = np.var(optimized_artwork_heatmap, ddof = 1)
    return variance #노멀라이즈 이전

def regularization_cost(optimized_wall_list):
    new_artwork_positions = []
    new_artwork_distance = []
    for wall in optimized_wall_list:
        new_artwork_positions.append(wall['hanged_artworks']['new_coords']) 
    for i in range(len(new_artwork_positions)):
        if i < len(new_artwork_positions) - 1:
            new_artwork_distance.append(math.dist(new_artwork_positions[i], new_artwork_positions[i+1])) 
        else: new_artwork_distance.append(math.dist(new_artwork_positions[i], new_artwork_positions[0])) 
    new_artwork_distance_arr = np.array(new_artwork_distance)
    variance = np.var(new_artwork_distance_arr, ddof = 1)
    return variance #노멀라이즈 이전

def similarity_cost(optimized_artwork_list):
    artist_list = ["한놈", "두식이", "석삼", "너구리", "오징어", "육개장"] #작가 수가 10 미만이라 그냥 리스트 만들고 진행, 나중에 돌리기 전에 지정
    artist_variance_list = []
    for artist in artist_list:
        same_artist_coords_list = []
        same_artist_dist_list = []

        for artwork in optimized_artwork_list:
            if artwork['artist'] == artist:
                same_artist_coords_list.append(artwork['new_coords'])

        for i in range(len(same_artist_coords_list)):
            if i < len(same_artist_coords_list) - 1:
                same_artist_dist_list.append(math.dist(same_artist_coords_list[i], same_artist_coords_list[i+1])) 
            else: same_artist_dist_list.append(math.dist(same_artist_coords_list[i], same_artist_coords_list[0])) 
        
        same_artist_dist_arr = np.array(same_artist_dist_list)
        same_artist_dist_var = np.var(same_artist_dist_arr, ddof = 1)
        artist_variance_list.append(same_artist_dist_var)

    variance_sum = sum(artist_variance_list)
    return variance_sum #노멀라이즈 이전


