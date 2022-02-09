import cv2
import numpy as np

img = cv2.imread("img/intersection.png")
height, width, channels = img.shape
img_output = np.zeros((height, width, 3), np.uint8)

# create masked black and white image where only the street is visible

# cv2 uses BRG, so we have to reverse the RGB colors
# color of lane marking (157, 234, 50)
lower_mask = np.array([145, 190, 40])
upper_mask = np.array([167, 255, 80])
masked_marking = cv2.inRange(img, lower_mask, upper_mask)

# color of the street (128, 64, 128)
lower_mask = np.array([118, 54, 118])
upper_mask = np.array([138, 74, 138])
masked_street = cv2.inRange(img, lower_mask, upper_mask)

masked_image = cv2.bitwise_or(masked_marking, masked_street)
masked_image = cv2.merge((masked_image, masked_image, masked_image))

concat_image = cv2.hconcat([img, masked_image])

cv2.imshow("", concat_image)
cv2.waitKey(0)
