import cv2
import numpy as np

# Read the image
image = cv2.imread('./no-bg.png')
cv2.imshow("First", image)

blurred = cv2.GaussianBlur(image, (0, 0), 10)

# Calculate the unsharp mask
unsharp_image = cv2.addWeighted(image, 2, blurred, -1, 0)
cv2.imshow("Second", unsharp_image)

# Convert the image to grayscale
gray = cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2GRAY)

# Apply contrast adjustment
alpha = 5  # Contrast control (1.0 for original image)
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)

# Display the original and adjusted images
cv2.imshow('Original', gray)
cv2.imshow('Adjusted', adjusted)
cv2.imwrite("temp1.jpg", adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()