import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pickle
import easygui

CUT = 1 #What portion of the video is encoded. 1 is the whole video, 2 is half, etc
FRAMESKIP = 30 #How many frames between each new image. By default, it encodes every 30th image.
videodata = cv2.VideoCapture(easygui.fileopenbox())

frames = int(videodata.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(videodata.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videodata.get(cv2.CAP_PROP_FRAME_HEIGHT))
buf = np.empty(((frames // CUT) // FRAMESKIP, height, width, 3), np.dtype('uint8'))

print("Total number of frames:", frames)
print("Number of frames in dataset:", frames // CUT // FRAMESKIP)
print("FPS:", videodata.get(cv2.CAP_PROP_FPS))
print("Height:", height)
print("Width:", width)

currentFramesCounter = 0
ret = True
while (currentFramesCounter < frames // CUT - FRAMESKIP and ret):
    ret, buf[currentFramesCounter // FRAMESKIP] = videodata.read()
    currentFramesCounter += FRAMESKIP
    videodata.set(cv2.CAP_PROP_POS_FRAMES, currentFramesCounter)


videodata.release()


pickle.dump(buf, open("images2_2.p", "wb"))


