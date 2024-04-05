import cv2
import numpy as np

image_path = "Images/new.png"
map_path = "Images/map.png"

#Load images
img_rgb = cv2.imread(map_path)
template = cv2.imread(image_path)

#Make them grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#sift detector
sift = cv2.SIFT_create()

#find keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(template_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(img_gray, None)

#matching flan
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

#matching descriptors
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

#test ratio
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

#find coords
template_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
img_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#calculate perspective
M, _ = cv2.findHomography(template_pts, img_pts, cv2.RANSAC)

#apply perspective
h, w = template_gray.shape
template_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
transformed_corners = cv2.perspectiveTransform(template_corners, M)

#find center of template image
center_x = int((transformed_corners[0][0][0] + transformed_corners[2][0][0]) / 2)
center_y = int((transformed_corners[0][0][1] + transformed_corners[2][0][1]) / 2)

print("Template image center coordinates in map image:", (center_x, center_y))
