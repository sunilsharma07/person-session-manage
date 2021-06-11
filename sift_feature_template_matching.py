import numpy as np
import cv2
import matplotlib.pyplot as plt

# train image
img = cv2.imread(
    '/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/person-session-manage/w1.jpeg',0)

img2 = img.copy()

template = cv2.imread(
    '/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/person-session-manage/train.jpeg',0)

w, h = template.shape[::-1]


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 4)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()

    cv2.imshow('window-1',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # PRINT The Coordinates
    print(min_val, max_val, min_loc, max_loc)
