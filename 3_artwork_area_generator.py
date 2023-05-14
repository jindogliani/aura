"""
- 2번에서의 히트맵을 1번의 매트릭스랑 비교하여 작품들의 예상 영역 추출 후 매트릭스로 저장
"""

import os

currentPath = os.getcwd()
visitorDataArtworkList = "preAURA_mmdd_MMDD_artworks"
# "preAURA_MMDD_MMDD_artworks" => AURA 이전의 mmdd 부터 MMDD 까지의 작품들 .npy 들을 보관하는 폴더명 

print("현재 위치:" + currentPath)
os.makedirs(currentPath+ "/" + visitorDataArtworkList, exist_ok= True)

