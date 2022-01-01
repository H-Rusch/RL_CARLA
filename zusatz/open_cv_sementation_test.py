import cv2
import numpy as np

img = cv2.imread("img/middel_section.png")
height, width, channels = img.shape
img_output = np.zeros((height, width, 3), np.uint8)

# create masked black and white image where only the street is visible

# cv2 uses BRG, so we have to reverse the RGB colors
# color of lane marking (157, 234, 50)
lower_mask = np.array([40, 224, 147])
upper_mask = np.array([60, 244, 167])
masked_marking = cv2.inRange(img, lower_mask, upper_mask)

# color of the street (128, 64, 128)
lower_mask = np.array([118, 54, 118])
upper_mask = np.array([138, 74, 138])
masked_street = cv2.inRange(img, lower_mask, upper_mask)

masked_image = cv2.bitwise_or(masked_marking, masked_street)

# find the contour with the largest area which is the street
contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

street, largest_area = None, 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > largest_area:
        largest_area = area
        street = contour

cv2.drawContours(img_output, street, -1, (255, 255, 255), 1)

concat_image = cv2.hconcat([img, img_output])
# concat_image = cv2.vconcat([concat_image, concat_image])

img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)

cv2.imshow("", concat_image)
cv2.waitKey(0)
