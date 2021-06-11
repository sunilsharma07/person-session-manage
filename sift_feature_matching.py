# have two image but same object how can identify
# https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/

# sift person matching github
# https://github.com/zszazi/OpenCV-Template-matching-and-SIFT

# scale invariant feature transform pytorch github
# https://github.com/OpenGenus/SIFT-Scale-Invariant-Feature-Transform#:~:text=Scale%20Invariant%2DFeature%20Transform%20(SIFT),-This%20repository%20contains&text=D.,these%20features%20for%20object%20recognition.

import numpy as np
import cv2
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 30

# initiate SIFT detector
sift = cv2.SIFT_create()


def bfmatcher():
    # BF(brute force)Matcher with default params
    bf = cv2.BFMatcher()
    return bf


def flannmatcher():
    # Create the Flann Matcher object
    FLANN_INDEX_KDITREE = 0
    flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
    flann = cv2.FlannBasedMatcher(flannParam, {})
    return flann


# train image
img = cv2.imread(
    '/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/person-session-manage/w2.jpeg')

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img, None)
train_img_kp = cv2.drawKeypoints(img, kp1, None, (255, 0, 0), 4)  # draw keypoints of the train image
plt.imshow(train_img_kp)  # show the train image keypoints
plt.title('Train Image Keypoints')
plt.show()

template = cv2.imread(
    '/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/person-session-manage/train.jpeg')
kp2, des2 = sift.detectAndCompute(template, None)

flann_obj = flannmatcher()
matches = flann_obj.knnMatch(des1, des2, k=2)

#bfmatcher_obj = bfmatcher()
#matches = bfmatcher_obj.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])


print(len(good))

img3 = cv2.drawMatchesKnn(img, kp1, template, kp2, good, None, flags=2)
cv2.imshow('image window2', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
