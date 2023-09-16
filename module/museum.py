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

class SceneActions(Enum):
    Forward = 0
    Swap = 1
    Flip = 2

class DataLoader():
    def __init__(self, artwork_data_path = "Daegu_new.json", exhibition_data_path = "Data_2022.json", wall_list_path = 'wall_list_2022.pkl', save_path = 'exhibited_artwork_list_2022.pkl'):
        with open(wall_list_path, 'rb') as f:
                wall_list = pickle.load(f)
        self.wall_list = wall_list
        if os.path.isfile(save_path):
            with open(save_path, 'rb') as f:
                self.exhibited_artwork_list = pickle.load(f)
        else:
            cwd = os.getcwd()
            # artwork_data_path 작품의 메타데이터가 있는 JSON 파일 => 작품의 size 값 추출
            # exhibition_data_path "Data_2022.json" #작품이 걸려있는 전시 내용 JSON 파일 => 작품의 positions 값 추출
            #2022년도 하정웅 미술관 벽 정보 리스트 로드

            #전체 작품 111개 리스트
            with open(artwork_data_path, "r") as f:
                artwork_data = json.load(f)
            artwork_list = [{"id": artwork["id"], "size": artwork["dimensions"]} for artwork in artwork_data["exhibitionObjects"]] #작품 id 리스트
            artwork_ids = [artwork["id"] for artwork in artwork_data["exhibitionObjects"]] #작품 id 리스트
            #전시된 작품 40개 리스트
            with open(exhibition_data_path, 'r', -1, encoding='utf-8') as f:
                exhibition_data = json.load(f)
            total_exhibited_artwork_list = [{"id": artwork["artworkIndex"], "pos_x": round(artwork["position"]["x"], 3), "pos_z": round(artwork["position"]["z"], 3)} for artwork in exhibition_data["paintings"]] #작품 id 리스트

            #2022년 기준 13작품만 관람객 데이터를 수집 함
            hall5_exhibited_artwork_list = ["PA-0064", "PA-0067", "PA-0027", "PA-0025", "PA-0087", "PA-0070", "PA-0066", "PA-0045", "PA-0079", "PA-0024", "PA-0085", "PA-0063", "PA-0083"]

            exhibited_artwork_list =[] #2022년 하정웅미술관 5전시실 최종 13개 작품만. 2023년도 올해는 작품 개수 달라질 예정
            for exhibited_artwork in total_exhibited_artwork_list:
                if exhibited_artwork["id"] in hall5_exhibited_artwork_list:
                    art_size = artwork_list[artwork_ids.index(exhibited_artwork["id"])]["size"]
                    exhibited_artwork["width"] = round(float(art_size.split("x")[1]) / 100, 3) #width 정보만 빼오기
                    exhibited_artwork["placed"] = False
                    # 더 좋은 방법이 없을까..... from 태욱
                    for wall in wall_list: #TODO
                        if wall["theta"] == 0: 
                            if (wall["x1"] <= exhibited_artwork["pos_x"] <= wall["x2"]) and (abs(wall["z1"] - exhibited_artwork["pos_z"]) <= 0.2):
                                exhibited_artwork["wall"] = wall["id"]
                                exhibited_artwork["theta"] = wall["theta"]
                        elif wall["theta"] == 180: 
                            if (wall["x2"] <= exhibited_artwork["pos_x"] <= wall["x1"]) and (abs(wall["z1"] - exhibited_artwork["pos_z"]) <= 0.2):
                                exhibited_artwork["wall"] = wall["id"]
                                exhibited_artwork["theta"] = wall["theta"]        
                        elif wall["theta"] == 90:
                            if (wall["z1"] <= exhibited_artwork["pos_z"] <= wall["z2"]) and (abs(wall["x1"] - exhibited_artwork["pos_x"]) <= 0.2):
                                exhibited_artwork["wall"] = wall["id"]
                                exhibited_artwork["theta"] = wall["theta"]   
                        elif wall["theta"] == -90:
                            if (wall["z2"] <= exhibited_artwork["pos_z"] <= wall["z1"]) and (abs(wall["x1"] - exhibited_artwork["pos_x"]) <= 0.2):
                                exhibited_artwork["wall"] = wall["id"]
                                exhibited_artwork["theta"] = wall["theta"]   
                        else:
                            continue #TODO #작품에 오일러앵글 쓰자 ! 올해는!!
                    exhibited_artwork_list.append(exhibited_artwork)
            #pickle로 저장.
            with open(save_path, 'wb') as f:
                pickle.dump(exhibited_artwork_list,f)
            self.exhibited_artwork_list = exhibited_artwork_list
        
        self._get_data()

    def _get_data(self):
        scene = {}
        artwork_data = {}
        wall_data = {}
        for wall in self.wall_list:
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

        for wall in self.wall_list:
            wall_data[wall['id']] = wall

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
            self.art_in_wall[wall_id] = sorted([art for art, (wall, pos) in self.scene_data.items() if wall == wall_id], key=lambda x: self.scene_data[x][1])

    def update_scene(self, scene_data):
        self.scene_data = scene_data
        for wall_id in self.wall_data.keys():
            self.art_in_wall[wall_id] = sorted([art for art, (wall, pos) in self.scene_data.items() if wall == wall_id], key=lambda x: self.scene_data[x][1])

    def get_legal_actions(self):
        possible_actions = []

        swap_possible_walls = {}
        for wall_id in self.wall_data.keys():
            wall_len = int(self.wall_data[wall_id]['length']*10) - 3
            swap_possible_walls[wall_id] = []
            if len(self.art_in_wall[wall_id]) == 0:
                swap_possible_walls[wall_id].append((0, wall_len))

            elif len(self.art_in_wall[wall_id]) >= 1:
                possible_actions.append((SceneActions.Flip, None , wall_id, 0))
                prev_end = 0
                for idx, art_in_wall_id in enumerate(self.art_in_wall[wall_id]):
                    try:
                        next_start = self.scene_data[self.art_in_wall[wall_id][idx+1]][1] - int(self.artwork_data[self.art_in_wall[wall_id][idx+1]]['width']/2)*10 - 3
                    except:
                        next_start = wall_len
                    art_len = int(self.artwork_data[art_in_wall_id]['width'])*10
                    pos = self.scene_data[art_in_wall_id][1]
                    if art_len % 2 != 0:
                        start = pos - int(art_len/2) - 3
                        end = pos + int(art_len/2)
                    else:
                        start = pos - int(art_len/2) - 3
                        end = pos + int(art_len/2) - 1
                    swap_possible_walls[wall_id].append((prev_end, start))
                    swap_possible_walls[wall_id].append((end, next_start - end))
                    prev_end = end
                
            
        for art_id in self.artwork_data.keys():
            wall_id = self.scene_data[art_id][0]
            wall_len = int(self.wall_data[wall_id]['length'])*10 - 3
            art_len = int(self.artwork_data[art_id]['width'])*10
            pos = self.scene_data[art_id][1]
            if wall_len - sum([self.artwork_data[art]['width']*10 for art in self.art_in_wall[wall_id]]) > 13:
                if len(self.art_in_wall[wall_id]) == 1:
                    if wall_len - pos - int(art_len/2) > 0:
                        possible_actions.append((SceneActions.Forward, art_id, None, wall_len - pos - int(art_len/2)))
                elif len(self.art_in_wall[wall_id]) > 1: 
                    try:
                        next_art_id = self.art_in_wall[self.art_in_wall[wall_id].index(art_id)+1]
                        next_pos = self.scene_data[next_art_id][1]
                        next_art_len = int(self.artwork_data[next_art_id]['width'])*10
                        remain_space = (next_pos - int(next_art_len/2) - 3) - (pos + int(art_len/2))
                        if remain_space > 0:
                            possible_actions.append((SceneActions.Forward, art_id, None, remain_space))
                    except:
                        if wall_len - pos - int(art_len/2) > 0:
                            possible_actions.append((SceneActions.Forward, art_id, None, wall_len - pos - int(art_len/2)))

            for possible_wall_id, possible_space in swap_possible_walls.items():
                if possible_wall_id != wall_id:
                    for (possible_start, possible_length) in possible_space:
                        if possible_length >= art_len + 3:
                            possible_actions.append((SceneActions.Swap, art_id, possible_wall_id, possible_start+3+int(art_len/2)))

            
        return possible_actions
    
    def do_action(self, action, art, wall, value):
        new_scene = self.scene_data
        if action == SceneActions.Flip:
            assert art == None
            for _art in self.art_in_wall[wall]:
                    _pos = self.scene_data[_art][1]
                    wall_len = self.wall_data[wall]['length']*10
                    if self.artwork_data[_art]['width'] % 2 != 0:
                        new_scene[_art] = (wall, wall_len - _pos)
                    else:
                        new_scene[_art] = (wall, wall_len - _pos)
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
        for k, v in self.scene_data.items():
            art = self.artwork_data[k]
            wall = self.wall_data[v[0]]
            pos = v[1]
            
            wall_width = int(wall['length']*10)
            art_len = int(art['width']*10)
            vis_list = [0] * wall_width
            if art_len % 2 != 0:
                vis_list[pos-int(art_len/2):pos+int(art_len/2)+1] = [1] * art_len
            else:
                vis_list[pos-int(art_len/2):pos+int(art_len/2)] = [1] * art_len

            if v[0] not in  draw.keys():
                draw[v[0]] = np.array(vis_list)
            else:
                draw[v[0]] += np.array(vis_list)
        
    def print_scene(self):
        self.draw = {}
        for k, v in self.scene_data.items():
            art = self.artwork_data[k]
            wall = self.wall_data[v[0]]
            pos = v[1]
            
            wall_width = int(wall['length']*10)
            art_len = int(art['width']*10)
            vis_list = [0] * wall_width
            if art_len % 2 != 0:
                vis_list[int(pos-int(art_len/2)):int(pos+int(art_len/2)+1)] = [1] * art_len
                vis_list[int(pos)] = 2
            else:
                vis_list[int(pos-int(art_len/2)):int(pos+int(art_len/2))] = [1] * art_len
                vis_list[int(pos)] = 2

            if v[0] not in  self.draw.keys():
                self.draw[v[0]] = np.array(vis_list)
            else:
                self.draw[v[0]] += np.array(vis_list)

        for k, vis_list in self.draw.items():
            vis_string = ''.join(map(str, vis_list))
            print(k + " : " + vis_string)

if __name__ == "__main__":
    scene = MuseumScene()
    legal_moves = scene.get_legal_actions()
    for _ in range(1):
        scene.print_scene()
        print("##################################################")
        action_tup = choice(legal_moves)
        print(action_tup)
        scene.do_action(SceneActions.Flip, None, 'w0', 0)
        scene.print_scene()


