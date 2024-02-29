import math

#figure용 샘플 중심점 값 구하기

same_artist_coords_list1 = [(7.381051, 12.78086), (9.073837, 16.1654), (13.82945, 20.23103), (16.23921, 19.01182)]
same_artist_coords_list2 = [(8.366638, 2.624134), (11.03151, 2.626517), (14.6999, 2.62253), (19.68547, -1.109945)]

same_artist_coords_list = same_artist_coords_list2
same_artist_dist_list = []


def centroid(coordinates):
    x_coords = [p[0] for p in coordinates]
    y_coords = [p[1] for p in coordinates]
    _len = len(coordinates)
    if _len == 0:
        return (0, 0)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return (centroid_x, centroid_y)

centroid_coord = centroid(same_artist_coords_list)

for coords in same_artist_coords_list:
    same_artist_dist_list.append(math.dist(coords, centroid_coord))

print(centroid_coord)
print(same_artist_dist_list)