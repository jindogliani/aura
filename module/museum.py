import copy
import csv
import json
import math
import os
import sys
sys.path.insert(0,os.path.join(os.getcwd(), 'module'))
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
from visualization import *

class SceneActions(Enum):
    Forward = 0
    Swap = 1
    Flip = 2
    Add = 3
    Delete = 4

ver = "2023"
heatmap_cell_size_cm = 10 #10cm로 셀 사이즈 지정

class DataLoader():
    def __init__(self, wall_list_path = '_wall_list_with_artworks.pkl', save_path = '_exhibited_artwork_list.pkl'):
        with open(ver + wall_list_path, 'rb') as f:
                wall_list = pickle.load(f)
        self.wall_list = wall_list
        if os.path.isfile(ver+save_path):
            with open(ver + save_path, 'rb') as f:
                self.exhibited_artwork_list = pickle.load(f)
        self._get_data() # scene data 구조체 생성

    def _get_data(self): # __init__ 에서 데이터 소환
        scene = {}
        artwork_data = {}
        wall_data = {}
        heatmap_data = {}
        if ver == '2022':
            heatmap_path = 'Data_2022_preAURA_2022+(9-27-19-59)/'
        elif ver == '2023':
            heatmap_path = 'Data_2023_preAURA_2023+(9-24-17-25)/'
        
        for wall in self.wall_list:
            if wall['displayable'] == True:
                wall_data[wall['id']] = wall  #탐색 가능한 벽만 남겨둠.

        for artwork in self.exhibited_artwork_list:
            wall = wall_data[artwork['wall']]
            art_p = [artwork['pos_x']*heatmap_cell_size_cm, artwork['pos_z']*heatmap_cell_size_cm]
            wall_0 = [wall['x1']*heatmap_cell_size_cm, wall['z1']*heatmap_cell_size_cm]
            wall_1 = [wall['x2']*heatmap_cell_size_cm, wall['z2']*heatmap_cell_size_cm]
            wall_vec = np.array([wall_1[0] - wall_0[0], wall_1[1] - wall_0[1]])
            art_vec = np.array([art_p[0] - wall_0[0], art_p[1] - wall_0[1]])
            theta = np.arccos(np.dot(wall_vec, art_vec) / (np.linalg.norm(wall_vec) * np.linalg.norm(art_vec)))
            art_pos = np.linalg.norm(art_vec) * np.cos(theta)
            art_pos = int(art_pos) #wall x1, z1�쓣 湲곗���쑝濡� artwork�쓽 �쐞移� �젙�닔媛�
            # 벽 안에서의 작품 위치를 int형으로... 난... 아직도 싫어...

            scene[artwork['id']] = (artwork['wall'], art_pos) #key: 작품 id, value: (벽 id, 벽 안에서의 작품 위치 int형)
            artwork_data[artwork['id']] = artwork
            artwork_visitor_heatmap = np.load(heatmap_path + artwork['id'] + '.npy') # cell size 10cm
            heatmap_data[artwork['id']] = np.sum(artwork_visitor_heatmap)

        self.scene_data = scene
        self.artwork_data = artwork_data
        self.wall_data = wall_data
        self.heatmap_data = dict(sorted(heatmap_data.items(), key=lambda x: x[1], reverse=False))

class MuseumScene():
    def __init__(self):
        dl = DataLoader()
        dl._get_data()
        self.scene_data = dl.scene_data
        self.origin_num = len(self.scene_data)
        self.artwork_data = dl.artwork_data
        self.wall_data = dl.wall_data
        self.heatmap_data = dl.heatmap_data
        self.worst_five_art = list(self.heatmap_data.keys())[:5]
        self.art_in_wall = {}
        self.artists = []
        for wall_id in self.wall_data.keys():
            self.art_in_wall[wall_id] = [] #wall 당 작품이 뭐뭐 들었나 확인
            self.art_in_wall[wall_id] = sorted([art for art, (wall, pos) in self.scene_data.items() if wall == wall_id], key=lambda x: self.scene_data[x][1])
        
        for art_id in self.scene_data.keys():
            tar_artist = self.artwork_data[art_id]['artist']
            if tar_artist not in self.artists:
                self.artists.append(tar_artist)
        self.origin_artist_num = len(self.artists)
        # print("Initial art and artisit number is %d and %d"%(self.origin_num, self.origin_artist_num))

        self.weights = {}
        g_weight = 0.9
        r_weight = 0.1
        s_weight = 0.0
        n_weight = 0.0
        an_weight = 0.0
        self.weights['g'] = g_weight
        self.weights['r'] = r_weight
        self.weights['s'] = s_weight
        self.weights['n'] = n_weight
        self.weights['an'] = an_weight
        
    def update_scene(self, scene_data):
        self.artists = []
        for art, (wall, pos) in scene_data.items():
            if type(pos) == float or pos < 0 or pos > int(self.wall_data[wall]['length']*heatmap_cell_size_cm):
                raise "fuck yourself"
            tar_artist = self.artwork_data[art]['artist']
            if tar_artist not in self.artists:
                self.artists.append(tar_artist)

        organized_scene_data = copy.deepcopy(scene_data)
        for wall_id in self.wall_data.keys():
            self.art_in_wall[wall_id] = []
            self.art_in_wall[wall_id] = sorted([art for art, (wall, pos) in scene_data.items() if wall == wall_id], key=lambda x: scene_data[x][1])
            
            if len(self.art_in_wall[wall_id]) > 1:
                art_list = sorted(self.art_in_wall[wall_id], key=lambda x: scene_data[x][1])
                self.art_in_wall[wall_id] = art_list

            elif len(self.art_in_wall[wall_id]) == 1:
                art_id = self.art_in_wall[wall_id][0]
                wall_middle = int(self.wall_data[wall_id]['length']*heatmap_cell_size_cm / 2)
                organized_scene_data[art_id] = (wall_id, wall_middle) #

        self.scene_data = organized_scene_data

    def get_legal_actions(self):
        possible_actions = {}
        # possible_actions = []
        Flip = []
        Forward = []
        Swap = []
        Delete = []
        Add = []
        swap_possible_walls = {}
            
        for wall_id in self.wall_data.keys():
            wall_len = int(self.wall_data[wall_id]['length']*10) - 3
            swap_possible_walls[wall_id] = []
            if len(self.art_in_wall[wall_id]) == 0:
                swap_possible_walls[wall_id].append((0, wall_len))

            elif len(self.art_in_wall[wall_id]) >= 1:
                Flip.append((SceneActions.Flip, None , wall_id, 0))
                prev_end = 0
                for idx, art_in_wall_id in enumerate(self.art_in_wall[wall_id]):
                    try:
                        _art_len = int(self.artwork_data[self.art_in_wall[wall_id][idx+1]]['width']*10/2) if int(self.artwork_data[self.art_in_wall[wall_id][idx+1]]['width']*10) % 2 == 0 else int(self.artwork_data[self.art_in_wall[wall_id][idx+1]]['width']*10/2) + 1
                        next_start = int(self.scene_data[self.art_in_wall[wall_id][idx+1]][1]) - _art_len- 3
                    except:
                        next_start = wall_len
                    
                    art_len = int(self.artwork_data[art_in_wall_id]['width']*10)
                    pos = int(self.scene_data[art_in_wall_id][1])
                    if art_len % 2 != 0:
                        art_len -= 1
                    start = pos - int(art_len/2) - 3
                    if start < 0:
                        start = 0
                    end = pos + int(art_len/2)

                    if type(start) == float or type(end) == float or type(prev_end) == float or type(next_start) == float:
                        raise None
                    if start < 0 or end < 0 or prev_end < 0 or next_start < 0:
                        raise None
                    
                    if prev_end == 0:
                        swap_possible_walls[wall_id].append((prev_end, start - prev_end))
                        swap_possible_walls[wall_id].append((end, next_start - end))
                        prev_end = end
                    else:
                        swap_possible_walls[wall_id].append((end, next_start - end))
                
            
        for art_id in self.artwork_data.keys():
            art_len = int(self.artwork_data[art_id]['width']*10)
            if art_len % 2 != 0:
                art_len -= 1
            if art_id in self.scene_data.keys():
                wall_id = self.scene_data[art_id][0]
                wall_len = int(self.wall_data[wall_id]['length']*10) - 3
                if art_id in self.worst_five_art:
                    Delete.append((SceneActions.Delete, art_id, None, None))
                pos = self.scene_data[art_id][1]
                if wall_len - sum([self.artwork_data[art]['width']*10 for art in self.art_in_wall[wall_id]]) > 3:
                    if len(self.art_in_wall[wall_id]) == 1:
                        if wall_len - (pos + int(art_len/2)) > 0:
                            Forward.append((SceneActions.Forward, art_id, None, wall_len - (pos + int(art_len/2))))
                    elif len(self.art_in_wall[wall_id]) > 1: 
                        try:
                            next_art_id = self.art_in_wall[wall_id][self.art_in_wall[wall_id].index(art_id)+1]
                            next_pos = self.scene_data[next_art_id][1]
                            next_art_len = int(self.artwork_data[next_art_id]['width']*10)
                            if next_art_len % 2 != 0:
                                next_art_len -= 1
                            remain_space = next_pos - pos - int(next_art_len/2) - (int(art_len/2))
                            if remain_space > 0:
                                Forward.append((SceneActions.Forward, art_id, None, remain_space))
                        except:
                            if wall_len - (pos + int(art_len/2)) > 0:
                                Forward.append((SceneActions.Forward, art_id, None, wall_len - (pos + int(art_len/2))))
                for possible_wall_id, possible_space in swap_possible_walls.items():
                    if possible_wall_id != wall_id:
                        while len(possible_space) >= 1:
                            possible_start, possible_length = possible_space.pop(0)
                            if possible_length >= art_len + 3:
                                Swap.append((SceneActions.Swap, art_id, possible_wall_id, possible_start+3+int(art_len/2)))
            else:
                for possible_wall_id, possible_space in swap_possible_walls.items():
                    if possible_wall_id != wall_id:
                        while len(possible_space) >= 1:
                            possible_start, possible_length = possible_space.pop(0)
                            if possible_length >= art_len + 3:
                                Add.append((SceneActions.Add, art_id, possible_wall_id, possible_start+3+int(art_len/2)))
        
        possible_actions["Flip"] = Flip
        possible_actions["Forward"] = Forward
        possible_actions["Swap"] = Swap
        possible_actions["Delete"] = Delete
        possible_actions["Add"] = Add
        return possible_actions
    
    def do_action(self, action, art, wall, value):
        new_scene = copy.deepcopy(self.scene_data)
        new_art_in_wall = copy.deepcopy(self.art_in_wall)
        if action == SceneActions.Flip:
            assert art == None
            for _art in self.art_in_wall[wall]:
                    _pos = self.scene_data[_art][1]
                    wall_len = int(self.wall_data[wall]['length']*10 )
                    if int(self.artwork_data[_art]['width']*10) % 2 == 0:
                        new_scene[_art] = (wall, wall_len - _pos)
                    else:
                        new_scene[_art] = (wall, wall_len - _pos - 1)
            for k, v in new_scene.items():
                if type(v[1]) == float or v[1] < 0 or v[1] > int(self.wall_data[v[0]]['length']*10):
                    raise "fuck your self"
            return new_scene
        
        if action == SceneActions.Swap:
            (past_wall, _) = self.scene_data[art]
            new_scene[art] = (wall, value)
            new_art_in_wall[wall].append(art)
            new_art_in_wall[past_wall].remove(art)
            for wall_id in [past_wall, wall]:
                if len(new_art_in_wall[wall_id]) > 1:
                    art_list = sorted(new_art_in_wall[wall_id], key=lambda x: new_scene[x][1])
                    new_art_in_wall[wall_id] = art_list
                    temp_len = 0
                    temp_ratio = 0
                    for i, art_in_wall in enumerate(art_list):
                        temp_len += self.artwork_data[art_in_wall]['width']
                    for art_in_wall in art_list:
                        coord_ratio = (round(temp_ratio, 3) + round(temp_ratio+self.artwork_data[art_in_wall]['width']/temp_len, 3)) / 2
                        new_pos = int(self.wall_data[wall_id]['length']*10 * coord_ratio)
                        new_scene[art_in_wall] = (wall_id, new_pos)
                        if new_pos == 0:
                            raise None
                        temp_ratio += self.artwork_data[art_in_wall]['width']/temp_len

                elif len(new_art_in_wall[wall_id]) == 1:
                    art_id = new_art_in_wall[wall_id][0]
                    wall_middle = int(self.wall_data[wall_id]['length']*10 / 2)
                    new_scene[art_id] = (wall_id, wall_middle)
                    
            return new_scene
        
        if action == SceneActions.Forward:
            assert wall == None
            _wall, _pos = self.scene_data[art]
            new_scene[art] = (_wall, _pos+value)
            for k, v in new_scene.items():
                if type(v[1]) == float or v[1] < 0 or v[1] > int(self.wall_data[v[0]]['length']*10):
                    raise "fuck yourself"
            return new_scene
        
        if action == SceneActions.Add:
            new_scene[art] = (wall, value)
            return new_scene

        if action == SceneActions.Delete:
            new_scene.pop(art)
            return new_scene
        
    def get_weights(self):
        return self.weights
        
        
    def evaluation(self):
        draw = {}
        g_weight = self.weights['g']
        r_weight = self.weights['r']
        s_weight = self.weights['s']
        n_weight = self.weights['n']
        an_weight = self.weights['an']

        g_cost = (1-goal_cost(self.scene_data, self.artwork_data, self.wall_data))
        r_cost = 1-regularization_cost(self.scene_data, self.artwork_data, self.wall_data)
        s_cost = 1-similarity_cost(self.scene_data, self.artwork_data, self.wall_data)
        # r_cost = 0
        # s_cost = 0
        # n_cost = 0
        # an_cost = 0
        n_cost = len(self.scene_data) / self.origin_num
        an_cost = len(self.artists) / self.origin_artist_num

        if g_cost < 0: g_cost = 0
        if r_cost < 0: r_cost = 0
        if s_cost < 0: s_cost = 0
        total_cost = g_weight * g_cost + r_weight * r_cost + s_weight * s_cost + n_weight * n_cost + an_cost * an_weight
        costs = [g_cost, r_cost, s_cost, n_cost, an_cost]
        # print(costs)

        return total_cost, costs

    def visualize(self, num):
        with open('Results/final.pickle', 'rb') as f:
            best_scene_data = pickle.load(f)
        
        visualization(best_scene_data, self.artwork_data, self.wall_data, num)
        # visualization(best_scene_data, self.artwork_data, self.wall_data, num)
        # convert_scene_json(best_scene_data, self.artwork_data, self.wall_data, num)
        # print(wall_list)
             
    # def print_scene(self):
    #     self.draw = {}
    #     for k, v in self.wall_data.items():
    #         vis_list = [0] * int(v['length']*10)
    #         self.draw[k] = np.array(vis_list)
        
    #     #print(self.scene_data)

    #     for k, v in self.scene_data.items():
    #         art = self.artwork_data[k]
    #         wall = self.wall_data[v[0]]
    #         pos = v[1]
            
    #         wall_width = int(wall['length']*10)
    #         art_len = int(art['width']*10)
    #         vis_list = [0] * wall_width
    #         assert len(vis_list) == wall_width
    #         if art_len % 2 != 0:
    #             # print(vis_list)
    #             vis_list[int(pos-int(art_len/2)):int(pos+int(art_len/2))+1] = [1] * art_len
    #             # print(vis_list)
    #             vis_list[int(pos)] = 5
    #             # print(vis_list)
    #             assert len(vis_list) == wall_width
    #         else:
    #             # print(vis_list)
    #             vis_list[int(pos-int(art_len/2)):int(pos+int(art_len/2))] = [1] * art_len
    #             # print(vis_list)
    #             vis_list[int(pos)] = 5
    #             # print(vis_list)
    #             assert len(vis_list) == wall_width

    #         self.draw[v[0]] += np.array(vis_list)

    #     for k, vis_list in self.draw.items():
    #         if 2 in vis_list:
    #             raise "KILL ME"
    #         vis_string = ''.join(map(str, vis_list))
    #         print(k + " : " + vis_string)

if __name__ == "__main__":
    scene = MuseumScene()
    
    # print(len(scene.scene_data))
    # print(scene.scene_data)

    # for k, v in scene.scene_data.items():
    #     print(k, v)
    
    idx = 000
    scene.visualize(idx)

    #scene.print_scene()
    
    # total_cost = scene.evaluation()
    # print(total_cost)
    # idx = 100
    # scene.visualize(idx)

    # print("=====================================")
    # for moves in legal_moves:
    #     print(moves)
    
    # sum_num = 0
    # for idx in range(1000):
    #     legal_moves_dict = scene.get_legal_actions() #dict
    #     legal_moves = []
    #     for v in legal_moves_dict.values():
    #         legal_moves += v
    #     sum_num += len(legal_moves)
    #     print(sum_num / (idx+1))
    #     action_tup = choice(legal_moves)
    #     # for idx, action_tup in enumerate(legal_moves):
    #     #     new_scene = scene.do_action(*action_tup)
    #     new_scene = scene.do_action(*action_tup)
    #     scene.update_scene(new_scene)
    #     # scene.visualize(new_scene, idx)