import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
CUT = 5
FRAMESKIP = 90
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
    ret, buf[currentFramesCounter // FRAMESKIP] = videodata.read()
    currentFramesCounter += FRAMESKIP
    videodata.set(cv2.CAP_PROP_POS_FRAMES, currentFramesCounter)


videodata.release()

for i in range(0, 60):

    img = buf[i]
    plt.imshow(img)
    plt.show()
    time.sleep(5)
    img = cv2.pyrDown(img)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imggray, (5, 5), 1)

    c = 255 / np.log(1 + np.max(imgblur))
    log_image = c * (np.log(imgblur + 1))

    # Specify the data type so that
    # float value will be converted to int
    log_image = np.array(log_image, dtype=np.uint8)
    bilateral = cv2.bilateralFilter(log_image, 15, 75, 75)
    t, th3 = cv2.threshold(log_image, 200, 255, cv2.THRESH_BINARY_INV)

    edge = cv2.Canny(image=bilateral, threshold1=50, threshold2=200)
    plt.imshow(edge,  cmap='gray')
    plt.show()
    time.sleep(5)


cv2.imwrite('image.png', img)
