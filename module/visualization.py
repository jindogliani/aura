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
print (best_scene_data)

ver = "2023"
heatmap_dict = {}

if ver == "2023":
    with open("Data_2023.json", 'r', -1, encoding='utf-8') as f:
        exhibition_data = json.load(f)
    with open('2023_wall_list_with_artworks.pkl', 'rb') as f:
        wall_list = pickle.load(f)
    with open('2023_exhibited_artwork_list.pkl', 'rb') as f:
        exhibited_artwork_list = pickle.load(f)

    space_heatmap = np.load('SpaceData/2023_GMA+(9-27-14-50).npy') #cell size 0.1
    space_heatmap[space_heatmap > 254] = -6 
    space_heatmap[space_heatmap == 0] = -3 
    space_heatmap[space_heatmap == 127] = 0 

    init_scene_data = {'PA-0023': ['w5', 28], 'PA-0026': ['w5', 65], 'KO-0009': ['w6', 29], 'PA-0095': ['w8', 12], 'PA-0098': ['w8', 33], 'PA-0076': ['w8', 49], 'PA-0074': ['w9', 16], 'PA-0075': ['w9', 31], 'PA-0077': ['w9', 49], 'KO-0010': ['w9', 80], 'KO-0008': ('w9', 116), 'PA-0101': ['w9', 132], 'PA-0057': ['w10', 11], 'PA-0052': ['w10', 29], 'PA-0061': ['w10', 49], 'PA-0001': ['w14', 33], 'PA-0003': ['w18', 41], 'PA-0004': ['w18', 93], 'PA-0082': ['w24', 17], 'PA-0084': ['w24', 48], 'PA-0083': ['w26', 21], 'PA-0063': ['w26', 46], 'PA-0067': ['w27', 22], 'PA-0064': ['w27', 49], 'PA-0024': ['w31', 53], 'PA-0087': ['w41', 28], 'PA-0027': ['w42', 16], 'PA-0025': ['w42', 46], 'PA-0036': ['w43', 19], 'PA-0085': ['w43', 40], 'PA-0086': ['w43', 55], 'PA-0070': ['w45', 12], 'PA-0065': ['w45', 50], 'PA-0031': ['w46', 35], 'PA-0088': ['w50', 34], 'PA-0100': ['w52', 29], 'PA-0099': ['w52', 58], 'KO-0007': ['w56', 33], 'KO-0006': ['w57', 49], 'KO-0004': ['w57', 86], 'KO-0005': ['w57', 112], 'PA-0090': ['w57', 154], 'PA-0089': ['w58', 34]}
    best_scene_data = init_scene_data #임시로 값 설정해놓고 

    for k, v in init_scene_data.items():
        artwork_visitor_heatmap = np.load('Data_2023_preAURA_2023+(9-24-17-25)/'+ k + '.npy') # cell size 10cm
        # artwork_visitor_heatmap = np.load('Daegu_new_preAURA_2023+(9-23)/'+ k + '.npy') # cell size 20cm
        heatmap_dict[k] = artwork_visitor_heatmap

if ver == "2022":
    with open("Data_2022.json", 'r', -1, encoding='utf-8') as f:
        exhibition_data = json.load(f)
    with open('2022_wall_list_with_artworks.pkl', 'rb') as f:
        wall_list = pickle.load(f)
    with open('2022_exhibited_artwork_list.pkl', 'rb') as f:
        exhibited_artwork_list = pickle.load(f)

    space_heatmap = np.load('SpaceData/2022_HJW+(9-27-20-52).npy') #cell size 0.1
    space_heatmap[space_heatmap > 254] = -6 
    space_heatmap[space_heatmap == 0] = -3 
    space_heatmap[space_heatmap == 127] = 0 

    for a in exhibited_artwork_list:
        artwork_visitor_heatmap = np.load('Data_2022_preAURA_2022+(9-27-19-59)/'+ a['id'] + '.npy') # cell size 10cm
        # artwork_visitor_heatmap = np.load('Daegu_new_preAURA_2023+(9-23)/'+ k + '.npy') # cell size 20cm
        heatmap_dict[a['id']] = artwork_visitor_heatmap


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

    if ver == "2023":
        x_offset, z_offset = 7, 12
    elif ver == "2022":
        x_offset, z_offset = 25, 20

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
        artwork_visitor_heatmap = heatmap_dict[k] # cell size 10cm
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
    
    with open("Data_"+ ver +"_optimized_%d.json"%num, "w") as f:
        json.dump(exhibition_data, f)
    