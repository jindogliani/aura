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
import json

with open('_best_scene_100_10000_10cm+(9-26-8-6).pickle', 'rb') as f:
    best_scene_data = pickle.load(f)

with open('_wall_list_2023.pkl', 'rb') as f:
    wall_list = pickle.load(f)

with open('_exhibited_artwork_list_2023.pkl', 'rb') as f:
    exhibited_artwork_list = pickle.load(f)

exhibition_data_path = "_Data_2023.json"
with open(exhibition_data_path, 'r', -1, encoding='utf-8') as f:
    exhibition_data = json.load(f)

# print (best_scene_data)

# space_heatmap = np.load('SpaceData/coords_GMA3+(9-23).npy') #cell size 0.2
space_heatmap = np.load('SpaceData/coords_GMA3+(9-24-17-21).npy') #cell size 0.1
space_heatmap[space_heatmap > 254] = -6 
space_heatmap[space_heatmap == 0] = -3 
space_heatmap[space_heatmap == 127] = 0 

space_vertical_size, space_horizontal_size = 40, 40
heatmap_cell_size = 0.1
space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
space_horizontal_cells, space_vertcal_cells = round(space_horizontal_cells), round(space_vertcal_cells)

def heatmap_generator(
    artwork_width, new_pos_x, new_pos_z, old_pos_x, old_pos_z, x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap
):
    x1, x2, z1, z2 = new_pos_x - artwork_width/2, new_pos_x + artwork_width/2, new_pos_z, new_pos_z
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) 
    
    direction = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    direction = cv2.line(direction, (_x1, _z1), (round((_x1+_x2)/2), round((_z1+_z2)/2)), 255, 1)
    direction_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 90, 0.7)
    direction = cv2.warpAffine(direction, direction_rotation, direction.shape)
    artwork_heatmap += direction

    artwork_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta, 1)
    artwork_heatmap = cv2.warpAffine(artwork_heatmap, artwork_rotation, artwork_heatmap.shape)
    artwork_heatmap[artwork_heatmap > 0] = -10 
    #artwork_heatmap[artwork_heatmap > 0] = 0 

    
    #artwork_visitor_rotation = cv2.getRotationMatrix2D((round(old_pos_x/heatmap_cell_size), round(old_pos_z/heatmap_cell_size)), 0, 1)
    #artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_visitor_transform = np.float32([[1, 0, round((new_pos_x - old_pos_x)/heatmap_cell_size)], [0, 1, round((new_pos_z - old_pos_z)/heatmap_cell_size)]])
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)
    artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta - old_theta, 1)
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_heatmap += artwork_visitor_heatmap
    return artwork_heatmap

def visualization(scene_data, artwork_data, wall_data, num):

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
        artwork_visitor_heatmap = np.load('Daegu_new_preAURA_2023+(9-24-17-25)/'+ k + '.npy') # cell size 10cm
        # artwork_visitor_heatmap = np.load('Daegu_new_preAURA_2023+(9-23)/'+ k + '.npy') # cell size 20cm
        artwork_heatmap = heatmap_generator(art["width"], new_art_pos[0], new_art_pos[1], old_art_pos[0], old_art_pos[1], x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap)
        scene_heatmap += artwork_heatmap
    
    scene_heatmap += space_heatmap

    heatmapCSV = pd.DataFrame(scene_heatmap)
    sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-10, vmax=50)
    plt.savefig('__visualization_%d.png'%num)
    plt.close()

def convert_scene_json(scene_data, artwork_data, wall_data, num):
    for k, v in scene_data.items():
        art = artwork_data[k]
        wall = wall_data[v[0]]
        pos = v[1]
        
        temp_heatmap_cell_size = 0.1
        ratio1, ratio2 = (wall["length"]-pos*temp_heatmap_cell_size)/wall["length"], (pos*temp_heatmap_cell_size)/wall["length"]
        new_art_pos = (wall["x1"]*ratio1+wall["x2"]*ratio2, wall["z1"]*ratio1+wall["z2"]*ratio2)
        new_theta = wall["theta"]

        for exhibited_artwork in exhibition_data["paintings"]:
            if art["id"] == exhibited_artwork["artworkIndex"]:
                exhibited_artwork["position"]["x"] = new_art_pos[0]
                exhibited_artwork["position"]["z"] = new_art_pos[1]
                exhibited_artwork["rotation"]["eulerAngles"]["y"] = new_theta
    
    with open("_Data_2023_optimized_%d.json"%num, "w") as f:
        json.dump(exhibition_data, f)
