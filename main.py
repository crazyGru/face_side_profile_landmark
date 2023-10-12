import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# Read the image
image = cv2.imread('./Source/f1.jpg', cv2.COLOR_BGR2GRAY)
height, width, _ = image.shape
image = cv2.resize(image, (width//2, height//2))

# blurred = cv2.GaussianBlur(image, (0, 0), 1)

# # Calculate the unsharp mask
# unsharp_image = cv2.addWeighted(image, 3, blurred, -1, 0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

mean_value = cv2.mean(gray)[0]
if mean_value > 127:
    gray = 255 - gray
    mean_value = 255 - mean_value

print(mean_value)

cv2.imshow('Original', image)
# cv2.imshow('Sharpened', unsharp_image)

alpha = 2  # Contrast control (1.0 for original image)
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)

# Apply thresholding (optional)
cv2.imshow("Gray", adjusted)
ret, thresh = cv2.threshold(adjusted, mean_value, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

height, width, _ = mask.shape
print(height, width)
temp = []
for i in range(height):
    for j in range(width):
        flag = False
        for k in range(3):
            if mask[i][j][k]:
                flag = True
        
        if flag:
            temp.append([i, j])
            break

contour_image = np.zeros((height, width, 3), np.uint8)

print(len(temp))

color = (0, 255, 0) 
thickness = 1

slash = height//200
print(slash)

features = []

for i in range(slash, len(temp)-slash):
    contour_image[temp[i][0]][temp[i][1]] = [255,255,255]

    if(temp[i][1]-temp[i-slash][1])*(temp[i+slash][1]-temp[i][1])==0:
        contour_image[temp[i][0]][temp[i][1]] = [255,0,0]
        cv2.circle(contour_image, (temp[i][1],temp[i][0]), 3, (0, 0, 255), 2)
        # i+=slash
        features.append([temp[i][0], temp[i][1]])
        


X = np.array(features)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=10, min_samples=1).fit(X)

# Get the cluster labels
labels = dbscan.labels_

# Create sets of points based on the cluster labels
point_sets = {}
for i, label in enumerate(labels):
    if label not in point_sets:
        point_sets[label] = []
    point_sets[label].append(features[i])


keys = point_sets.keys()

print(keys)
noised_removed_features = []
for key in point_sets.keys():
    x = 0
    y = 0
    for temp in point_sets[key]:
        x = x + temp[0]
        y = y + temp[1]
    
    x = x // len(point_sets[key])
    y = y // len(point_sets[key])
    noised_removed_features.append([x, y])

for i in range(len(noised_removed_features)):
    cv2.circle(contour_image, (noised_removed_features[i][1], noised_removed_features[i][0]), 2, (255,255,0), 2)
    cv2.circle(result, (noised_removed_features[i][1], noised_removed_features[i][0]), 2, (255,0,0), 2)
    cv2.putText(result, str(i), (noised_removed_features[i][1], noised_removed_features[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

##########################   MATCHING   STEP     ############################

file_name = "./Match/f1.pts"  # Replace with your file name

try:
    with open(file_name, 'r') as file:
        file_content = file.read()
except FileNotFoundError:
    print(f"File '{file_name}' not found.")

points = []
start_index = file_content.index('{') + 1
end_index = file_content.index('}')
points_data = file_content[start_index:end_index].split('\n')

for point_data in points_data:
    if point_data.strip() != '':
        x, y = map(float, point_data.strip().split())
        cv2.circle(result, (int(x/2), int(y/2)), 2, (255,0,255), 2)
        points.append((x, y))

points = sorted(points, key=lambda p: p[1])

print(points)

threshold = 0.5

distance = cdist(noised_removed_features, points)

cv2.imshow("finla", result)
cv2.imshow("contour", contour_image)
cv2.imwrite("result.jpg", image)
cv2.waitKey(0)