import cv2
import numpy as np

# Read the image
image = cv2.imread('./Source/f1.jpg')

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (0, 0), 3)

# Calculate the unsharp mask
unsharp_image = cv2.addWeighted(image, 3, blurred, -1, 0)

# Display the original and sharpened images
cv2.imshow('Original', image)
cv2.imshow('Sharpened', unsharp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()