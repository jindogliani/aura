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
        return self.exhibited_artwork_list, self.wall_list

if __name__ == "__main__":
    dataloader = DataLoader()
    exhibited_artwork_list, wall_list = dataloader.get_data()
    print("end")
