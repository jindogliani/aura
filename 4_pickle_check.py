import os
from time import localtime, time
import json
import math
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('2023_wall_list_with_artworks.pkl', 'rb') as f:
    wall_list_23 = pickle.load(f)

for wall in wall_list_23:
    print(wall)

print()

with open('2023_exhibited_artwork_list.pkl', 'rb') as f:
    exhibited_artwork_list_23 = pickle.load(f)

for artwork in exhibited_artwork_list_23:
    print(artwork)

print()
print()
print()

'''
with open('2022_wall_list_with_artworks.pkl', 'rb') as f:
    wall_list_22 = pickle.load(f)

for wall in wall_list_22:
    print(wall)

print()

with open('2022_exhibited_artwork_list.pkl', 'rb') as f:
    exhibited_artwork_list_22 = pickle.load(f)

for artwork in exhibited_artwork_list_22:
    print(artwork)

print(len(exhibited_artwork_list_22))
'''

print(len(exhibited_artwork_list_23))