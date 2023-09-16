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

from MCTS import MCTS, Node
from collections import namedtuple
from random import choice


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

    def get_data(self):
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

        return self.wall_data, self.artwork_data, self.scene_data
    
_SCENE = namedtuple("MuseumScene", "scene wall pos terminal")
class MuseumScene(_SCENE, Node):
    def find_children(self):
        #get all action results
        if self.terminal:
            return set()
        return {
            self.make_move(i) for i in range(4)
        }
    
    def find_random_child(self):
        #choose random action
        if self.terminal:
            return None
        return self.make_move(choice(range(4)))
    
    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        else:
            reward = 0
            for k, v in self.draw.items():
                uniq, cont = np.unique(v, return_counts=True)
                for num, count in zip(uniq, cont):
                    if num == 1:
                        reward += count
                    elif num > 1:
                        reward -= count * num

            for k, v in self.scene.items():
                art = self.artwork[k]
                wall = self.wall[v[0]]
                pos = v[1]
                if pos+art['width']/2 > wall['length'] or pos-art['width']/2 < 0:
                    reward -= 100
                else:
                    reward += 10
            for wall in self.wall:
                wall_art = [exhibited_art for exhibited_art, v in self.scene.items() if v[0] == wall]
                total_length = 0
                if len(wall_art) > 1:
                    for exhibited_art in wall_art:
                        total_length += self.artwork[exhibited_art]['width']
                
                if total_length > self.wall[wall]['length']:
                    reward -= 100

            return reward
                

    def apply_move(self):
        self.draw = {}
        for k, v in self.scene.items():
            art = self.artwork[k]
            wall = self.wall[v[0]]
            pos = v[1]
            
            wall_width = int(wall['length']*10)
            art_len = int(art['width']*10)
            vis_list = [0] * wall_width
            if art_len % 2 != 0:
                vis_list[pos-int(art_len/2):pos+int(art_len/2)+1] = [1] * art_len
            else:
                vis_list[pos-int(art_len/2):pos+int(art_len/2)] = [1] * art_len

            if v[0] not in self.draw.keys():
                self.draw[v[0]] = np.array(vis_list)
            else:
                self.draw[v[0]] += np.array(vis_list)
    
    def is_terminal(self):
        return self.terminal
    
    def choose_target(self):
        #0 pos + 1(random art) #1 pos - 1(random art) #2 flip all(random wall) #3 change wall(random art)
        self.random_art = choice(list(self.artwork.keys()))
        self.random_wall = choice(list(self.wall.keys()))
    
    def make_move(self, idx):
        new_scene = copy.deepcopy(self.scene)
        #TODO do action
        if idx == 0:
            #Action 1. move forward
            wall, pos = self.scene[self.random_art]
            new_scene[self.random_art] = [wall, pos+1]
        elif idx == 1:
            #Action 2. move backward
            wall, pos = self.scene[self.random_art]
            new_scene[self.random_art] = [wall, pos-1]
        elif idx == 2:
            #Action 3. flip all
            wall_len = self.wall[self.random_wall]['length']
            for k, v in self.scene.items():
                wall, pos = v
                if wall == self.random_wall:
                    new_scene[k] = [wall, wall_len - pos]
        elif idx == 3:
            #Action 4. change wall
            wall_art = [exhibited_art for exhibited_art, v in self.scene.items() if v[0] == self.random_wall]
            target_art = choice(wall_art)
            target_wall, target_pos = self.scene[target_art]
            origin_wall, origin_pos = self.scene[self.random_art]
            new_scene[self.random_art] = [target_wall, target_pos]
            new_scene[target_art] = [origin_wall, origin_pos]
        else:
            raise RuntimeError(f"Invalid action {idx}")
        
        is_terminal = self.terminal
        new_museum = MuseumScene(new_scene, self.wall, self.artwork, is_terminal)
        new_museum.choose_target()
        return new_museum

    def print_scene(self):
        self.apply_move()
        for k, vis_list in self.draw.items():
            vis_string = ''.join(map(str, vis_list))
            print(k + " : " + vis_string)



def new_museum_scene():
    init_data = DataLoader()
    scene_data = init_data.get_scene()
    arts = []
    walls = []
    pos = []
    for k, v in scene_data.items():
        arts.append(k)
        walls.append(v[0])
        pos.append(v[1])
    new_musuem =  MuseumScene(
        art=arts,
        wall=walls,
        pos=pos,
        terminal=False
    )
    print(hash(new_musuem))
    new_musuem.choose_target()
    return new_musuem
    
def reorganize():
    tree = MCTS()
    museum = new_museum_scene()
    museum.print_scene()
    iter = 0
    while True:
        #TODO do action and roll out
        # museum = museum.make_move()
        if iter > 100:
            museum.terminal = True
        iter += 1
        if museum.terminal:
            break
        for _ in range(50):
            print("Training...")
            tree.do_rollout(museum)
        museum = tree.choose(museum)
        museum.print_scene()
        if museum.terminal:
            break

if __name__ == "__main__":
    init_data = DataLoader()
    wall_data, art_data, scene_data = init_data.get_data()
    art_in_wall = []
    for wall_id in wall_data.keys():
        art_in_wall = [art for art, (wall, pos) in scene_data.items() if wall == wall_id]
        print(len(art_in_wall))
