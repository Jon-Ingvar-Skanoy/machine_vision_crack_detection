import numpy as np
import cv2
import matplotlib.pyplot as plt

CUT = 30
FRAMESKIP = 30
videodata = cv2.VideoCapture(r"C:\Users\jonin\Desktop\machine vision\20230926_130609.mp4")
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
    ret, buf[currentFramesCounter // 30] = videodata.read()
    currentFramesCounter += FRAMESKIP
    videodata.set(cv2.CAP_PROP_POS_FRAMES, currentFramesCounter)

videodata.release()

img = buf[23]
cv2.imwrite('image.png', img)
