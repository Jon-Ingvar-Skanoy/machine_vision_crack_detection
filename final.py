import numpy as np
import cv2
import copy
import pickle
import tqdm
import time
import easygui
import datetime
import os

 #pickle.load(open(filepath, "rb"))
downscale = 4
cracks = 0
FRAMESKIP = 30

def weight_center(image):

    pass

def readVideo(inputFile):
    CUT = 1
    videodata = cv2.VideoCapture(inputFile)

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
        #print(currentFramesCounter / FRAMESKIP)

    videodata.release()
    return buf

def brownFilter(image, th3, i):
    brownFilterInput = copy.deepcopy(image)
    brownFilterInput = cv2.pyrDown(brownFilterInput)
    brownFilterInput = cv2.pyrDown(brownFilterInput)

    brown_lower = (57,73,119)
    brown_upper = (94,109,154)

    mask1 = cv2.inRange(brownFilterInput,brown_lower,brown_upper)
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.bitwise_not(cv2.dilate(mask1,kernel, iterations=10))

    return cv2.bitwise_and(th3,mask)

def whiteFilter(image, th3):
    whiteFilterInput = copy.deepcopy(image)
    whiteFilterInput = cv2.pyrDown(whiteFilterInput)
    whiteFilterInput = cv2.pyrDown(whiteFilterInput)

    white_lower = (200, 200, 200)
    white_upper = (255, 255, 255)

    mask1 = cv2.inRange(whiteFilterInput, white_lower, white_upper)
    kernal = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask1, kernal, iterations=10)
   # mask = removeTooSmall(mask, 300)

    mask = cv2.bitwise_not(mask)

    return cv2.bitwise_and(th3, mask), mask

def removeTooSmall(image, limit):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8, cv2.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]

    image = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= limit:  # keep
            image[labels == i + 1] = 255
    return image

def bikeFilter(image):
    for i in range(int(400 / downscale), int(1080 / downscale)):
        for j in range(int(1300 / downscale), int(1900 / downscale)):
            image[i][j] = 0
    return image


def blur_image(image):
    img_blur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.pyrDown(img_blur)
    img_blur = cv2.pyrDown(img_blur)

    img_blur = cv2.GaussianBlur(img_blur, (9, 9), 100000000000)
    img_blur = cv2.medianBlur(img_blur, 9)
    return img_blur

def remove_to_strait_lines(image):
    linesP = cv2.HoughLinesP(image, 0.6, 7 * np.pi / 180, 45, None, 90, 5)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0, 0, 0), 30, cv2.LINE_AA)
    return image
i = 0

filepath = easygui.fileopenbox()

if filepath.endswith('p'):
    IMGS = pickle.load(open(filepath, "rb"))
else:
    IMGS = readVideo(filepath)

if not os.path.isdir("detected_cracks"):
    os.makedirs("detected_cracks")
f = open(r"detected_cracks/detected cracks.txt", "w")
f.close()
f = open(r"detected_cracks/detected cracks.txt", "a")
crackIndexList = []
for img in tqdm.tqdm(IMGS):

    image_blur = blur_image(img)

    c = 255 / np.log(1 + np.max(image_blur))
    log_image = c * (np.log(image_blur + 1))
    log_image = np.array(log_image, dtype=np.uint8)


    th3 = cv2.adaptiveThreshold(log_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,1.1)

    th3 = removeTooSmall(th3, 100)

    th3 = remove_to_strait_lines(th3)

    th3 = removeTooSmall(th3, 100)

    result, mask = whiteFilter(img,th3)
    result = brownFilter(img,th3,i)
    result = bikeFilter(result)
    result = removeTooSmall(result, 110)

    score = np.sum(result) / 255

    if score > 1000:
        crackIndexList.append(i)
        if i-1 in crackIndexList:
            f.write(f"\n        Another subsequent crack found immediately after the previous")
        else:
            f.write(f"\nCracks detected at {datetime.timedelta(seconds=(i*30)//FRAMESKIP)}\n    Score: {score},    Index: {i}")
            cv2.imwrite(f'detected_cracks/image{i}.png', img)
            cv2.imwrite(f'detected_cracks/image{i}_TH3_{score}.png', result)
              #  cv2.imwrite(f'detected_cracks/{i}image_mask.png', mask)
                #cv2.imwrite(f'detected_cracks/image_blur{i}.png', image_blur)
              #  cv2.imwrite(f'detected_cracks/image_log{i}.png',log_image)
               # cv2.imwrite(f'detected_cracks/image_th{i}.png', th3)
            time.sleep(0.1)
            cracks +=1
    i+=1
f.close()
print(cracks)