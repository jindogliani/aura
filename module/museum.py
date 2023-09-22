import copy
import csv
import json
import math
import os
from time import localtime, time
import time

import pickle
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from random import choice
from enum import Enum
from cost import *

class SceneActions(Enum):
    Forward = 0
    Swap = 1
    Flip = 2

class DataLoader():
    def __init__(self, artwork_data_path = "Daegu_new.json", exhibition_data_path = "Data_2023.json", wall_list_path = 'wall_list_2023.pkl', save_path = 'exhibited_artwork_list_2023.pkl'):
        with open(wall_list_path, 'rb') as f:
                wall_list = pickle.load(f)
        self.wall_list = wall_list
        if os.path.isfile(save_path):
            with open(save_path, 'rb') as f:
                self.exhibited_artwork_list = pickle.load(f)
        self._get_data()

    def _get_data(self):
        scene = {}
        artwork_data = {}
        wall_data = {}
        for wall in self.wall_list:
            if wall['displayable'] == True:
                wall_data[wall['id']] = wall

        for artwork in self.exhibited_artwork_list:
            wall = wall_data[artwork['wall']]
            art_p = [artwork['pos_x']*10, artwork['pos_z']*10]
            wall_0 = [wall['x1']*10, wall['z1']*10]
            wall_1 = [wall['x2']*10, wall['z2']*10]
            wall_vec = np.array([wall_1[0] - wall_0[0], wall_1[1] - wall_0[1]])
            art_vec = np.array([art_p[0] - wall_0[0], art_p[1] - wall_0[1]])
            theta = np.arccos(np.dot(wall_vec, art_vec) / (np.linalg.norm(wall_vec) * np.linalg.norm(art_vec)))
            art_pos = np.linalg.norm(art_vec) * np.cos(theta)
            art_pos = int(art_pos) #wall x1, z1을 기준으로 artwork의 위치 정수값

            scene[artwork['id']] = [artwork['wall'], art_pos]
            artwork_data[artwork['id']] = artwork

        self.scene_data = scene
        self.artwork_data = artwork_data
        self.wall_data = wall_data

class MuseumScene():
    def __init__(self):
        dl = DataLoader()
        dl._get_data()
        self.scene_data = dl.scene_data
        self.artwork_data = dl.artwork_data
        self.wall_data = dl.wall_data
        self.art_in_wall = {}
        for wall_id in self.wall_data.keys():
            self.art_in_wall[wall_id] = []
            self.art_in_wall[wall_id] = sorted([art for art, (wall, pos) in self.scene_data.items() if wall == wall_id], key=lambda x: self.scene_data[x][1])

    def update_scene(self, scene_data):
        self.scene_data = scene_data
        for wall_id in self.wall_data.keys():
            self.art_in_wall[wall_id] = sorted([art for art, (wall, pos) in self.scene_data.items() if wall == wall_id], key=lambda x: self.scene_data[x][1])

    def get_legal_actions(self):
        possible_actions = []

        swap_possible_walls = {}
        for wall_id in self.wall_data.keys():
            wall_len = int(self.wall_data[wall_id]['length']*10) - 3 -1
            swap_possible_walls[wall_id] = []
            if len(self.art_in_wall[wall_id]) == 0:
                swap_possible_walls[wall_id].append((0, wall_len))

            elif len(self.art_in_wall[wall_id]) >= 1:
                possible_actions.append((SceneActions.Flip, None , wall_id, 0))
                prev_end = 0
                for idx, art_in_wall_id in enumerate(self.art_in_wall[wall_id]):
                    try:
                        next_start = self.scene_data[self.art_in_wall[wall_id][idx+1]][1] - int(self.artwork_data[self.art_in_wall[wall_id][idx+1]]['width']*10/2) - 3
                    except:
                        next_start = wall_len
                    art_len = int(self.artwork_data[art_in_wall_id]['width']*10)
                    pos = self.scene_data[art_in_wall_id][1]
                    if art_len % 2 != 0:
                        start = pos - int(art_len/2) - 3
                        end = pos + int(art_len/2) + 1
                    else:
                        start = pos - int(art_len/2) - 3
                        end = pos + int(art_len/2)
                    swap_possible_walls[wall_id].append((prev_end, start - prev_end))
                    swap_possible_walls[wall_id].append((end, next_start - end))
                    prev_end = end
                
            
        for art_id in self.artwork_data.keys():
            wall_id = self.scene_data[art_id][0]
            wall_len = int(self.wall_data[wall_id]['length']*10) - 3 - 1
            art_len = int(self.artwork_data[art_id]['width']*10)
            pos = self.scene_data[art_id][1]
            if wall_len - sum([self.artwork_data[art]['width']*10 for art in self.art_in_wall[wall_id]]) > 13:
                if len(self.art_in_wall[wall_id]) == 1:
                    if wall_len - (pos + int(art_len/2) if art_len %2 != 0 else pos + int(art_len/2) - 1) > 0:
                        possible_actions.append((SceneActions.Forward, art_id, None, wall_len - (pos + int(art_len/2) if art_len %2 != 0 else pos + int(art_len/2) - 1)))
                elif len(self.art_in_wall[wall_id]) > 1: 
                    try:
                        next_art_id = self.art_in_wall[wall_id][self.art_in_wall[wall_id].index(art_id)+1]
                        next_pos = self.scene_data[next_art_id][1]
                        next_art_len = int(self.artwork_data[next_art_id]['width']*10)
                        remain_space = next_pos - pos - int(next_art_len/2) - (int(art_len/2) if art_len %2 != 0 else int(art_len/2) - 1) - 3
                        if remain_space > 0:
                            possible_actions.append((SceneActions.Forward, art_id, None, remain_space))
                    except:
                        if wall_len - (pos + int(art_len/2) if art_len %2 != 0 else pos + int(art_len/2) - 1) > 0:
                            possible_actions.append((SceneActions.Forward, art_id, None, wall_len - (pos + int(art_len/2) if art_len %2 != 0 else pos + int(art_len/2) - 1)))

            for possible_wall_id, possible_space in swap_possible_walls.items():
                if possible_wall_id != wall_id:
                    while len(possible_space) >= 1:
                        possible_start, possible_length = possible_space.pop(0)
                        if possible_length >= art_len - 1 + 3:
                            possible_actions.append((SceneActions.Swap, art_id, possible_wall_id, possible_start+3+int(art_len/2)))
                            break
            
        return possible_actions
    
    def do_action(self, action, art, wall, value):
        new_scene = self.scene_data
        if action == SceneActions.Flip:
            assert art == None
            for _art in self.art_in_wall[wall]:
                    _pos = self.scene_data[_art][1]
                    wall_len = self.wall_data[wall]['length']*10 
                    if self.artwork_data[_art]['width'] % 2 == 0:
                        new_scene[_art] = (wall, wall_len - _pos)
                    else:
                        new_scene[_art] = (wall, wall_len - 1 - _pos)
            return new_scene
        
        if action == SceneActions.Swap:
            new_scene[art] = (wall, value)
            return new_scene
        
        if action == SceneActions.Forward:
            assert wall == None
            _wall, _pos = self.scene_data[art]
            new_scene[art] = (_wall, _pos+value)
            return new_scene
        
    def evaluation(self):
        draw = {}

        g_weight = 1.0
        r_weight = 0.0
        s_weight = 0.4

        g_cost = goal_cost(self.scene_data, self.artwork_data, self.wall_data)
        r_cost = 0
        s_cost = 0
        # r_cost = regularization_cost(self.scene_data, self.artwork_data, self.wall_data)
        # s_cost = similarity_cost(self.scene_data, self.artwork_data, self.wall_data)

        total_cost = g_weight * g_cost + r_weight * r_cost + s_weight * s_cost

        costs = [g_cost, r_cost, s_cost] 
        return total_cost
                    
    def print_scene(self):
        self.draw = {}
        for k, v in self.wall_data.items():
            vis_list = [0] * int(v['length']*10)
            self.draw[k] = np.array(vis_list)
        
        #print(self.scene_data)

        for k, v in self.scene_data.items():
            art = self.artwork_data[k]
            wall = self.wall_data[v[0]]
            pos = v[1]
            
            wall_width = int(wall['length']*10)
            art_len = int(art['width']*10)
            vis_list = [0] * wall_width
            assert len(vis_list) == wall_width
            if art_len % 2 != 0:
                # print(vis_list)
                vis_list[int(pos-int(art_len/2)):int(pos+int(art_len/2))+1] = [1] * art_len
                # print(vis_list)
                vis_list[int(pos)] = 5
                # print(vis_list)
                assert len(vis_list) == wall_width
            else:
                # print(vis_list)
                vis_list[int(pos-int(art_len/2)):int(pos+int(art_len/2))] = [1] * art_len
                # print(vis_list)
                vis_list[int(pos)] = 5
                # print(vis_list)
                assert len(vis_list) == wall_width

            
            self.draw[v[0]] += np.array(vis_list)

        for k, vis_list in self.draw.items():
            if 2 in vis_list:
                raise "KILL ME"
            vis_string = ''.join(map(str, vis_list))
            print(k + " : " + vis_string)

if __name__ == "__main__":
    scene = MuseumScene()
    scene.print_scene()
    
    total_cost = scene.evaluation()
    print(total_cost)

    print("=====================================")
    # for moves in legal_moves:
    #     print(moves)
    # for idx in range(1):
    #     legal_moves = scene.get_legal_actions()
    #     action_tup = choice(legal_moves)
    #     new_scene = scene.do_action(*action_tup)
    #     scene.update_scene(new_scene)
    #     print("************************************")
    #     scene.print_scene()