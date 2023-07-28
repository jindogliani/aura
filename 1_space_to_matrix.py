# Unity 상에서 공간 캡처 데이터를 불러와서 매트릭스로 변형 및 .csv로 저장

import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path, threshold_value=254):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_threshold = cv2.threshold(
        image_grey, threshold_value, 255, cv2.THRESH_BINARY
    )
    image_threshold = cv2.bitwise_not(image_threshold)
    contours, _ = cv2.findContours(
        image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return image, image_grey, image_threshold, contours


gallery_image_path = "GalleryImage/GMA3_W-10.png"
wall_image_path = "GalleryImage/GMA3_W+0.5.png"

(
    gallery_image,
    gallery_image_grey,
    gallery_image_threshold,
    gallery_contours,
) = process_image(gallery_image_path)
(
    wall_image,
    wall_image_grey,
    wall_image_threshold,
    wall_contours,
) = process_image(wall_image_path)


# list comprehension
contours = [
    wall_contour
    for wall_contour in wall_contours
    if cv2.contourArea(wall_contour) > 3000
]

contour = contours[0]
mask = np.zeros(gallery_image_grey.shape, np.uint8)
cv2.drawContours(mask, [contour], -1, 255, 2)

# 컨투어 픽셀값 뽑기 => 실패
pixel_np = np.transpose(np.nonzero(mask))
pixel_cv = cv2.findNonZero(mask)
img = gallery_image.copy()
cv2.drawContours(img, [pixel_cv], -1, (0, 0, 255), 1)
cv2.imshow("numpy", mask)

gallery_image = cv2.drawContours(gallery_image, gallery_contours, -1, (0, 255, 0), 2)
cv2.imshow("Image", gallery_image)

# plt.figure(1)
# plt.imshow(cv2.cvtColor(wImageThreshold2, cv2.COLOR_GRAY2RGB))
# plt.figure(2)
# plt.imshow(cv2.cvtColor(imageThreshold, cv2.COLOR_GRAY2RGB))

plt.show()
cv2.waitKey(0)
cv2.imwrite("GalleryImage/test.png", gallery_image_threshold)
