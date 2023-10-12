import cv2
import numpy as np

image = cv2.imread('./Source/f2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=1)
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Foreground', result)
cv2.waitKey(0)
cv2.destroyAllWindows()