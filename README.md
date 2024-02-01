# AURA developer's notes

# Daegu_new == 2022, 2023년에 전시 했던 작품들 정보를 포함하고 있는 작품 목록 데이터 (작품 가로폭 사용)

# Data_2022 == 2022년 하정웅미술관 갤러리3,4,5에서 전시했던 전시 데이터 (작품목록, 위치 사용)
# Data_2023 == 2023년 광주시립미술관 갤러리6에서 전시했던 전시 데이터 (작품목록, 위치 사용)

# 1.py로 생성
# 2022_wall_list == 하정웅미술관 갤러리3,4,5 벽들의 공간 정보가 들어 있는 딕셔너리 리스트
# 2023_wall_list == 광주시립미술관 갤러리6 벽들의 공간 정보가 들어 있는 딕셔너리 리스트
# SpaceData/2022_HJW+(9-30-16-10).npy == 하정웅미술관 npy, 벽 안쪽 전시 공간 값 127, 벽 부분은 255
# SpaceData/2023_GMA+(9-27-14-50).npy == 광주시립미술관 npy, 벽 안쪽 전시 공간 값 127, 벽 부분은 255

# 2.py로 생성
# Data_2022_preAURA_2022+(9-27-19-59) == 2022년 전시 데이터 + 관람객 데이터 + 생성 날짜 (작품별 관람객 npy, 전체히트맵 csv 저장)
# Data_2023_preAURA_2023+(9-24-17-25) == 2023년 전시 데이터 + 관람객 데이터 + 생성 날짜 (작품별 관람객 npy, 전체히트맵 csv 저장)

# 3.py로 생성
# 2022_wall_list_with_artworks.pkl == 걸린 작품이 순서대로 정렬된 벽 딕셔너리 리스트
# 2023_wall_list_with_artworks.pkl == 걸린 작품이 순서대로 정렬된 벽 딕셔너리 리스트
# 2022_exhibited_artwork_list.pkl == 한붓그리기 형태로 정렬된 순서의 작품 딕셔너리 리스트
# 2023_exhibited_artwork_list.pkl == 한붓그리기 형태로 정렬된 순서의 작품 딕셔너리 리스트

# 4.py 는 딕셔너리 리스트 체크

# 5.py
# 각 2022, 2023 버전 별 initial Cost Function 및 initial score 각각 계산
# goal_cost() => cell들의 분산값 계산.
# init_cell_variance => initial cell 분산값
# regularization_cost() => Scene Rationality를 목적으로 한붓그리기 형태로 작품들 각각의 포지션 및 거리 계산 후, 거리에 대한 분산 측정
# init_distance_variance => 초기 작품 거리들의 분산값
# similarity_cost() => 같은 작가의 클러스터를 생성하고 클러스터 안에 centroid 값을 구하고 작품 포지션과의 거리 계산 후 작가별 거리 분산값을 계산 
# init_WCSS => 작가별 거리 분산값들의 합. WCSS (Within-Cluster Sum of Squares), 클러스터들의 거리분산의 합.

# 6.py
# 