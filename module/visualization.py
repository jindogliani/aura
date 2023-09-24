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

with open('best_scene_415.pickle', 'rb') as f:
    best_scene_data = pickle.load(f)

with open('wall_list_2023.pkl', 'rb') as f:
    wall_list = pickle.load(f)

with open('exhibited_artwork_list_2023.pkl', 'rb') as f:
    exhibited_artwork_list = pickle.load(f)

# print (best_scene_data)

space_heatmap = np.load('SpaceData/coords_GMA3+(9-23).npy')
space_heatmap[space_heatmap > 254] = -6 #���� ���� -10���� ��ȯ
space_heatmap[space_heatmap == 0] = -3 #���� �ܺ� ���� 0���� -15���� ��ȯ
space_heatmap[space_heatmap == 127] = 0 #���� ���� ���� 127���� 0���� ��ȯ

space_vertical_size, space_horizontal_size = 40, 40
heatmap_cell_size = 0.2
space_vertcal_cells, space_horizontal_cells = space_vertical_size / heatmap_cell_size, space_horizontal_size / heatmap_cell_size
space_horizontal_cells, space_vertcal_cells = round(space_horizontal_cells), round(space_vertcal_cells)

def heatmap_generator(
    artwork_width, new_pos_x, new_pos_z, old_pos_x, old_pos_z, x_offset, z_offset, heatmap_cell_size, old_theta, new_theta, artwork_visitor_heatmap
):
    #��ǰ ��ü indicate heatmap ����
    x1, x2, z1, z2 = new_pos_x - artwork_width/2, new_pos_x + artwork_width/2, new_pos_z, new_pos_z
    _x1, _x2, _z1, _z2 = ((x1 + x_offset)/heatmap_cell_size), (x2 + x_offset)/heatmap_cell_size, (z1 + z_offset)/heatmap_cell_size, (z2 + z_offset)/heatmap_cell_size
    _x1, _x2, _z1, _z2 = round(_x1), round(_x2), round(_z1), round(_z2)
    artwork_heatmap = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    artwork_heatmap = cv2.line(artwork_heatmap, (_x1, _z1), (_x2, _z2), 255, 1) #��ǰ ������ �β� 10cm
    
    #��ǰ�� ���ϰ� �ִ� ���� ǥ��
    direction = np.zeros((space_vertcal_cells, space_horizontal_cells), dtype = np.int16)
    direction = cv2.line(direction, (_x1, _z1), (round((_x1+_x2)/2), round((_z1+_z2)/2)), 255, 1)
    direction_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), 90, 0.7)
    direction = cv2.warpAffine(direction, direction_rotation, direction.shape)
    artwork_heatmap += direction

    #��ǰ ���ο� �� �������� ȸ��
    artwork_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta, 1)
    artwork_heatmap = cv2.warpAffine(artwork_heatmap, artwork_rotation, artwork_heatmap.shape)
    artwork_heatmap[artwork_heatmap > 0] = -10 # Ȯ�ο�
    #artwork_heatmap[artwork_heatmap > 0] = 0 # �л� ����

    #��ǰ ������ ��Ʈ�� ȸ�� #TODO
    
    #artwork_visitor_rotation = cv2.getRotationMatrix2D((round(old_pos_x/heatmap_cell_size), round(old_pos_z/heatmap_cell_size)), 0, 1)
    #artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_visitor_transform = np.float32([[1, 0, round((new_pos_x - old_pos_x)/heatmap_cell_size)], [0, 1, round((new_pos_z - old_pos_z)/heatmap_cell_size)]])
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_transform, artwork_visitor_heatmap.shape)
    artwork_visitor_rotation = cv2.getRotationMatrix2D((round((_x1+_x2)/2), round((_z1+_z2)/2)), new_theta - old_theta, 1)
    artwork_visitor_heatmap = cv2.warpAffine(artwork_visitor_heatmap, artwork_visitor_rotation, artwork_visitor_heatmap.shape)

    artwork_heatmap += artwork_visitor_heatmap
    return artwork_heatmap

def visualization(scene_data, artwork_data, wall_data, num):
    # ������ ���� �� �ְ� ���� �ʿ�
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

    heatmapCSV = pd.DataFrame(scene_heatmap)
    sns.heatmap(heatmapCSV, cmap='RdYlGn_r', vmin=-10, vmax=50)
    plt.savefig('visualization_%d.png'%num)
    plt.close()

if __name__ == "__main__":
    visualization(best_scene_data, exhibited_artwork_list, wall_list, 600)