import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

import requests

response = requests.post(
    'https://api.remove.bg/v1.0/removebg',
    files={'image_file': open('./temp3.jpg', 'rb')},
    data={'size': 'auto'},
    headers={'X-Api-Key': 'k2XDaxdXet3NUy7WX34njqSJ'},
timeout=60)
if response.status_code == requests.codes.ok:
    with open('no-bg.png', 'wb') as out:
        out.write(response.content)
else:
    print("Error:", response.status_code, response.text)


# Read the image
image = cv2.imread('./temp1.jpg', cv2.COLOR_BGR2GRAY)
saving_image = image.copy()
height, width = image.shape
image = cv2.resize(image, (width, height))

# blurred = cv2.GaussianBlur(image, (0, 0), 1)

# # Calculate the unsharp mask
# unsharp_image = cv2.addWeighted(image, 3, blurred, -1, 0)

# Convert the image to grayscale
gray = image.copy()

mean_value = cv2.mean(gray)[0]
# if mean_value > 127:
#     gray = 255 - gray
#     mean_value = 255 - mean_value

print(mean_value)

cv2.imshow('Original', image)
# cv2.imshow('Sharpened', unsharp_image)

alpha = 2  # Contrast control (1.0 for original image)
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)

# Apply thresholding (optional)
cv2.imshow("Gray", adjusted)
cv2.imwrite("1.jpg", adjusted)

ret, thresh = cv2.threshold(adjusted, 250, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(image.shape[:2], np.uint8)
for contour in contours:
    cv2.drawContours(mask, [contour], 0, (255), -1)
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite('2.jpg', result)

# Iterate over contours and draw them on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

cv2.imwrite('3.jpg', image)

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
    cv2.imshow("Largest Object", mask)
    cv2.imwrite("4.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found in the image")

cv2.waitKey(0)

height, width = mask.shape
print(height, width)
temp = []
for i in range(height):
    for j in range(width):
            # print(mask[i][j], i, j)
            if mask[i][j]:                
                temp.append([i, j])
                break

contour_image = np.zeros((height, width, 3), np.uint8)


color = (0, 255, 0) 
thickness = 1

slash = height//200

features = []

for i in range(slash, len(temp)-slash):
    contour_image[temp[i][0]][temp[i][1]] = [255,255,255]

    if(temp[i][1]-temp[i-slash][1])*(temp[i+slash][1]-temp[i][1])==0:
        contour_image[temp[i][0]][temp[i][1]] = [255,0,0]
        cv2.circle(contour_image, (temp[i][1],temp[i][0]), 3, (0, 0, 255), 2)
        # i+=slash
        features.append([temp[i][1], temp[i][0]])
        


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
    cv2.circle(contour_image, (noised_removed_features[i][0], noised_removed_features[i][1]), 2, (255,255,0), 2)
    # cv2.circle(result, (noised_removed_features[i][0], noised_removed_features[i][1]), 2, (255,0,0), 2)
    # cv2.putText(result, str(noised_removed_features[i][0]) +','+ str(noised_removed_features[i][1]), (noised_removed_features[i][0], noised_removed_features[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    # cv2.putText(result, str(i), (noised_removed_features[i][0], noised_removed_features[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

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


for i, point_data in enumerate(points_data):
    if point_data.strip() != '':
        x, y = map(float, point_data.strip().split())
        # cv2.circle(result, (int(x/2), int(y/2)), 2, (255,255,255), 2)
        # cv2.putText(result, str(int(x/2)) +','+ str(int(y/2)), (int(x/2), int(y/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness)
        # cv2.putText(result, str(int(x/2)) +','+ str(int(y/2)), (int(x/2), int(y/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness)
        points.append([x, y, i])


points = sorted(points, key=lambda p: p[1])


###############################################################################################

distances = cdist(noised_removed_features, [row[0:2] for row in points])
# distances = cdist(array1, array2)

# print(distances)
first_short_array = []
for i, distance in enumerate(distances):
    minx = min(distance)
    first_short_array.append([i, np.argmin(distance), minx])
    # print(i, ":", minx, noised_removed_features[i], points[np.argmin(distance)], ":", np.argmin(distance))
    # print(distance)
    # print("-----------------------------------")

first_short_array = sorted(first_short_array, key=lambda p: p[2])
###############################################################################################

distances = cdist([row[0:2] for row in points], noised_removed_features)
# distances = cdist(array1, array2)

# print(distances)
second_short_array = []
for i, distance in enumerate(distances):
    minx = min(distance)
    second_short_array.append([i, np.argmin(distance), minx])

second_short_array = sorted(second_short_array, key=lambda p: p[2])

# print(first_short_array)
# print(second_short_array)

#######################   SET  FINAL   POINTS   #####################################
final_points = np.zeros((len(second_short_array),3))
first_flags = np.zeros((1, len(first_short_array)))
second_flags = np.zeros((1, len(second_short_array)))

final_indexes = []

for final_point in first_short_array:
    if final_point[2] > 35:
        break
    if first_flags[0][final_point[0]] or second_flags[0][final_point[1]]:
        continue
    first_flags[0][final_point[0]] = 1
    second_flags[0][final_point[1]] = 1
    final_points[final_point[1]] = [noised_removed_features[final_point[0]][0], noised_removed_features[final_point[0]][1], points[final_point[1]][2]]
    final_indexes.append(final_point[1])

final_points[len(points)-1] = points[len(points)-1]
final_indexes.append(len(points)-1)

final_indexes = sorted(final_indexes)


for i in range(len(final_indexes)-1):
    min_val_y = final_points[final_indexes[i]][1]
    max_val_y = final_points[final_indexes[i+1]][1]
    current_range = max_val_y - min_val_y
    prev_min_val_y = points[final_indexes[i]][1]
    prev_max_val_y = points[final_indexes[i+1]][1]
    prev_range = prev_max_val_y - prev_min_val_y
    ratio = current_range / prev_range
    # print(min_val_y, max_val_y, ":", prev_min_val_y, prev_max_val_y, ":", "------------", final_indexes[i], final_indexes[i+1])
    # print("_________")
    for j in range(final_indexes[i]+1, final_indexes[i+1]):
        final_points[j][1] = min_val_y + (points[j][1]-prev_min_val_y)*ratio
        final_points[j][0] = points[j][0]
        final_points[j][2] = points[j][2]


final_points = sorted(final_points, key=lambda p:p[2] )
final_points[26][0] = final_points[10][0]
final_points[26][1] = final_points[3][1]

for i, final_point in enumerate(final_points):
    cv2.circle(saving_image, (int(final_point[0]), int(final_point[1])), 2, (255,255,0), 2)
    # print(final_point[2])
    cv2.putText(saving_image, str(int(30+final_point[2]-1)), (int(final_point[0]), int(final_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), thickness)

cv2.imshow("finla", result)
cv2.imwrite("5.jpg", result)
cv2.imshow("contour", contour_image)
cv2.imwrite("6.jpg", contour_image)
cv2.imwrite("result.jpg", saving_image)
cv2.waitKey(0)
