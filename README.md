# aura developer's notes

# 2023/08/05 홍진석

# 1_space_to_matrix.py 는 2022년 하정웅미술관 공간 좌표를 계산함
# 1_space_to_matrix_2023.py 이 2023년도 광주시립미술관 전체 공간 좌표를 계산함
# 1.1_space_to_matrix_NotInUse.py => 이전에 공간 캡처 이미지를 통해서 공간 매트릭스를 생성하던 방법임. 현재는 정확성이 떨어져서 사용하지 않음.

# 2023/08/07 홍진석

# 2_csv_to_heatmap.py => 작품 데이터: Daegu_new.json, 관람객 데이터: preAURA_1025_1030. 관람객 데이터 y_coords 나중에 음수값으로 반전 시켜야 함. 오프셋 값도 재지정 필요.
# 2.1_csv_to_heatmap_NotInUse.py => 음수값으로 반전된 형태. 사용할 때는 NotInUse 폴더에서 꺼내야 함.

# 3_artwork_area_generator.py