import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np

C2_list = [1,3,8,11,13,40,17,22,24,26,29,35,38,39]
C3_list = [2, 10, 12, 15, 18, 20, 23, 25, 27, 30, 32, 33, 36]

def create_heatmap(folder_path, base_image):
    all_data = []

    for participant_num in C3_list:
        csv_filename = f'playerLog_p{participant_num}.csv'
        csv_path = os.path.join(folder_path, csv_filename)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_data.append(df)

    if not all_data:
        print("No data found.")
        return

    # 전체 데이터 결합
    all_data_df = pd.concat(all_data)

    with Image.open(base_image) as img:
        img_width, img_height = img.size

        # 데이터 변환
        plot_width, plot_height = img_width * 4 / 5, img_height * 4 / 5
        center_x, center_y = img_width / 2, img_height / 2
        all_data_df['plot_x'] = center_x + (all_data_df['X'] / 10) * plot_width
        all_data_df['plot_z'] = center_y - (all_data_df['Z'] / 5) * plot_height

        # 히트맵 생성 - 색상 변경 및 밴드폭 조절
        plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
        sns.kdeplot(
            x=all_data_df['plot_x'], y=all_data_df['plot_z'],
            cmap="crest",  # 색상 변경
            fill=True, bw_adjust=0.5,  # 밴드폭 조절 (값 감소)
            alpha=0.6  # 히트맵 투명도 조정
        )

        # 이미지를 넘파이 배열로 변환
        img_array = np.array(img)

        plt.imshow(img_array, aspect='auto')

        # 축 제거 및 이미지 저장
        plt.axis('off')
        plt.savefig(os.path.join(folder_path, 'heatmap_C2.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()

# 사용 예시
folder_path = './playLog/Save_v2/'
base_image = './playLog/Save_v2/BaseFigure.jpg'
create_heatmap(folder_path, base_image)