import cv2
import numpy as np

# Load the original image
image = cv2.imread('./Source/f1.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the difference between Red and grayscale values
diff_image = gray_image.copy()  # Create a copy of the original image

# Loop through each pixel (i, j) in the image
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        red_value = image[i, j, 2]  # Red component (channel 2)
        gray_value = (image[i, j, 0] + image[i, j, 1] + image[i, j, 2]) /3
        diff = red_value - gray_value
        # Set the difference in the blue channel (channel 0)
        diff_image[i, j] = diff

# Save or display the difference image
cv2.imwrite('difference_image.jpg', diff_image)
cv2.imshow('Difference Image', diff_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



contours, hierarchy = cv2.findContours(diff_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(image.shape[:2], np.uint8)
for contour in contours:
    cv2.drawContours(mask, [contour], 0, (255), -1)
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite('result.jpg', result)

# Iterate over contours and draw them on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

if len(contours) > 0:
    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty mask
    mask = np.zeros_like(image)
    
    # Draw the largest contour on the mask
    cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), thickness=cv2.FILLED)
    
    # Bitwise-AND the mask with the original image to isolate the largest object
    result = cv2.bitwise_and(image, mask)
    
    # Display the result
    cv2.imshow("Largest Object", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found in the image")
