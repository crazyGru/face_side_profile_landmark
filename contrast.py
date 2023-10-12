import cv2
import numpy as np

# Read the image
image = cv2.imread('./Source/f1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply contrast adjustment
alpha = 2  # Contrast control (1.0 for original image)
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)

# Display the original and adjusted images
cv2.imshow('Original', gray)
cv2.imshow('Adjusted', adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()