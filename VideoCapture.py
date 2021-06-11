import numpy as np
import cv2
import pafy
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

url = "https://www.youtube.com/watch?v=MWqolax21Mc"

def get_youtube_cap(url):
    # play = pafy.new(url).streams[-1] # we will take the lowest quality stream
    # assert play is not None # makes sure we get an error if the video failed to load
    # return cv2.VideoCapture(play.url)
    cap = cv2.VideoCapture('/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/person-session-manage/Govinda.mp4')
    return cap


# Function to extract frames
def FrameCapture(vidObj):
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    while success:
        # vidObj object calls read
        # function extract frames

        vidObj.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))

        success, image = vidObj.read()

        # Saves the frames with frame-count
        cv2.imwrite("/media/sunil/06930e3e-f4e4-4037-bee5-327c2551e897/Downloads/kernal/kernal_scripts/person-session-manage/frame/frame%d.jpg" % count, image)
        count += 1


cap = get_youtube_cap(url)
FrameCapture(cap)


# ret, frame = cap.read()
#
# print(ret)
#
#
# cap.release()
#
# plt.imshow(frame[:, :, ::-1])  # OpenCV uses BGR, whereas matplotlib uses RGB
# plt.show()