# 색칠한 이미지로 주어졌다고 할 때
# 먼저 좌표값 추출하고 거기에 속성 부여 필요

import cv2
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from time import sleep

def background(file_num):
 
    # 이미지 읽어서 그레이스케일 변환, 바이너리 스케일 변환
    img = cv2.imread("GalleryImage/"+file_num+".png")
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, th = cv2.threshold(imgray, 7,255,cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(imgray, ksize=(11, 11), sigmaX=1)
    #ret, thresh1 = cv2.threshold(blur, 207, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blur, 10, 250)

    # 컨튜어 찾기
    contours, hr = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contr = contours[0]
    if file_num == "15":
        contr = contours[1]
        rect = cv2.minAreaRect(contr)
        (center_x, center_y), (w, h), rotation = rect
        vertex = cv2.boxPoints(rect)
        box = cv2.boxPoints(rect)   # 중심점과 각도를 4개의 꼭지점 좌표로 변환
        box = np.intp(box)          # 정수로 변환

        center_x = np.intp(center_x)
        center_y = np.intp(center_y)
        #print("thisis center", center_x, center_y)
        w = np.intp(w)
        h = np.intp(h)
        #print("thisis img", img[center_x][center_y])    # Oriented Bounding Box
    else:
        for contr in contours:  
            rect = cv2.minAreaRect(contr)
            (center_x, center_y), (w, h), rotation = rect
            vertex = cv2.boxPoints(rect)
            box = cv2.boxPoints(rect)   # 중심점과 각도를 4개의 꼭지점 좌표로 변환
            box = np.intp(box)          # 정수로 변환

            center_x = np.intp(center_x)
            center_y = np.intp(center_y)
            #print("thisis center", center_x, center_y)
            w = np.intp(w)
            h = np.intp(h)
            #print("thisis img", img[center_x][center_y])
        
    degree = ((np.arctan2(vertex[3][0]-vertex[0][0], vertex[3][1]-vertex[0][1])*180)/np.pi)
    #print("this is degree : ", degree)
    rotate_degree = cv2.getRotationMatrix2D((int(center_x), int(center_y)), -degree, 1)
    img_rotate = cv2.warpAffine(img, rotate_degree, (1000, 1000))

    #cv2.imshow("d", img_rotate)

    # new Image
    imgray = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2GRAY)
    #ret, th = cv2.threshold(imgray, 7,255,cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(imgray, ksize=(1, 1), sigmaX=0)
    #ret, thresh1 = cv2.threshold(blur, 207, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blur, 10, 250)
    #cv2.imshow('Edged', edged)
    #cv2.waitKey(0)    # 컨튜어 찾기

    contours, hr = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contr = contours[0]
    if file_num == "15":
        rect = cv2.minAreaRect(contr)
        (center_x, center_y), (w, h), rotation = rect
        vertex = cv2.boxPoints(rect)
        #print(vertex)
        box = cv2.boxPoints(rect)   # 중심점과 각도를 4개의 꼭지점 좌표로 변환
        box = np.intp(box)    
            
        center_x = np.intp(center_x)
        center_y = np.intp(center_y)
        w = np.intp(w)
        h = np.intp(h)
        #print(w, h)
        #cv2.drawContours(img_rotate, [box], -1, (220, 20, 20), 2)
    # Oriented Bounding Box
    
    else:
        for contr in contours:    
            #print(contr)
            #rect = cv2.minAreaRect(contr)
            #(center_x, center_y), (w, h), rotation = rect
            #vertex = cv2.boxPoints(rect)
        
            rect = cv2.minAreaRect(contr)
            (center_x, center_y), (w, h), rotation = rect
            vertex = cv2.boxPoints(rect)
            #print(vertex)
            box = cv2.boxPoints(rect)   # 중심점과 각도를 4개의 꼭지점 좌표로 변환
            box = np.intp(box)    
                
            center_x = np.intp(center_x)
            center_y = np.intp(center_y)
            w = np.intp(w)
            h = np.intp(h)
            #print(w, h)
            #cv2.drawContours(img_rotate, [box], -1, (220, 20, 20), 2)


    #cv2.imshow('rotate', img_rotate)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

   
    new_img = []
    #print(int(vertex[0][0]), int(vertex[0][1]), int(w), int(h))

    row1 = int(min(vertex[0][0], vertex[2][0]))
    row2 = int(max(vertex[0][0], vertex[2][0]))
    col1 = int(min(vertex[0][1], vertex[2][1]))
    col2 = int(max(vertex[0][1], vertex[2][1]))

    #print(row1, row2, col1, col2)
    new_img = img_rotate[col1:col2, row1:row2]

    #cv2.imshow('final', new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return new_img

    #x,y,w,h = cv2.boundingRect(contr)
    #cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 3)

    #cv2.imshow(W'Contour', img)
    #cv2.imwrite('edited_.png', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

"""
    # 이미지 로드 후 RGB로 변환
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # 사각형 좌표: 시작점의 x,y  ,넢이, 너비
    rectangle = (0, 0, 999, 999)

    # 초기 마스크 생성
    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    # grabCut에 사용할 임시 배열 생성
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # grabCut 실행
    cv2.grabCut(image_rgb, # 원본 이미지
            mask,       # 마스크
            rectangle,  # 사각형
            bgdModel,   # 배경을 위한 임시 배열
            fgdModel,   # 전경을 위한 임시 배열 
            5,          # 반복 횟수
            cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화
    # 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

    # 이미지에 새로운 마스크를 곱행 배경을 제외
    image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

    # plot
    plt.imshow(image_rgb_nobg)
    plt.show()

   """
def img_to_data(file_num):

    # Read img, tranlate grayscale, binaryscale
    #img = cv2.imreadfilename)
    img = background(file_num)
    img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    pixel_data = {'whole':0, 'movable': 0, 'sittable': 0, 'placeable': 0, 'distractor' : 0, 'black' : 0}
    
    # color 정리
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y][0] == 255:
                if img[x][y][1] == 255:
                    img[x][y] = (255, 255, 255)
                    pixel_data['movable'] += 1
                else:
                    img[x][y] = (255, 0, 0)
                    pixel_data['sittable'] += 1
            elif img[x][y][1] == 255:
                img[x][y] = (0, 255, 0)
                pixel_data['distractor'] += 1
            elif img[x][y][2] == 255:
                img[x][y] = (0, 0, 255)
                pixel_data['placeable'] += 1
            else:
                img[x][y] = (0, 0, 0)
                pixel_data['black'] += 1
    pixel_data['whole'] = pixel_data['movable'] + pixel_data['sittable'] + pixel_data['placeable'] + pixel_data['distractor'] + pixel_data['black']

    # 결과 출력
    cv2.imshow('Contour', img)
    ##cv2.imwrite('obb_' + file_num, img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return img, pixel_data


img_to_data('15')