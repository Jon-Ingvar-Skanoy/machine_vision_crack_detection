import numpy as np
import cv2
import copy
import pickle
import tqdm
import time
import easygui
import datetime
import os

# pickle.load(open(filepath, "rb"))
downscale = 4
cracks = 0
FRAMESKIP = 30 #How many frames between each new image. By default, it encodes every 30th image.


def read_video(input_file):
    cut = 1 #What portion of the video is encoded. 1 is the whole video, 2 is half, etc
    video_data = cv2.VideoCapture(input_file)

    frames = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty(((frames // cut) // FRAMESKIP, height, width, 3), np.dtype('uint8'))

    print("Total number of frames:", frames)
    print("Number of frames in dataset:", frames // cut // FRAMESKIP)
    print("FPS:", video_data.get(cv2.CAP_PROP_FPS))
    print("Height:", height)
    print("Width:", width)

    current_frames_counter = 0
    ret = True
    while current_frames_counter < frames // cut - FRAMESKIP and ret:
        ret, buf[current_frames_counter // FRAMESKIP] = video_data.read()
        current_frames_counter += FRAMESKIP
        video_data.set(cv2.CAP_PROP_POS_FRAMES, current_frames_counter)
        # print(current_frames_counter / FRAMESKIP)

    video_data.release()
    return buf


def brown_filter(image, th3):
    brown_filter_input = copy.deepcopy(image)
    brown_filter_input = cv2.pyrDown(brown_filter_input)
    brown_filter_input = cv2.pyrDown(brown_filter_input)

    brown_lower = (57, 73, 119)
    brown_upper = (94, 109, 154)

    mask1 = cv2.inRange(brown_filter_input, brown_lower, brown_upper)
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.bitwise_not(cv2.dilate(mask1, kernel, iterations=10))

    return cv2.bitwise_and(th3, mask)

def green_filter(image, th3):
    green_filter_input = copy.deepcopy(image)
    green_filter_input = cv2.pyrDown(green_filter_input)
    green_filter_input = cv2.pyrDown(green_filter_input)

    green_lower = (73,141,119)#(20,71,34)
    green_upper = (112,165,146)#(91,125,99)

    mask1 = cv2.inRange(green_filter_input, green_lower, green_upper)
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.bitwise_not(cv2.dilate(mask1, kernel, iterations=10))
    return cv2.bitwise_and(th3, mask)

def white_filter(image, th3):
    white_filter_input = copy.deepcopy(image)
    white_filter_input = cv2.pyrDown(white_filter_input)
    white_filter_input = cv2.pyrDown(white_filter_input)

    white_lower = (200, 200, 200)
    white_upper = (255, 255, 255)

    mask1 = cv2.inRange(white_filter_input, white_lower, white_upper)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask1, kernel, iterations=10)
    # mask = removeTooSmall(mask, 300)

    mask = cv2.bitwise_not(mask)

    return cv2.bitwise_and(th3, mask)

def colorFilter(image, th3):
    result = brown_filter(image,th3)
    result = green_filter(image,result)
    result = white_filter(image,result)
    return result


def remove_too_small(image, limit):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8, cv2.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]

    image = np.zeros(labels.shape, np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= limit:  # keep
            image[labels == i + 1] = 255
    return image


def bike_filter(image):
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


def remove_too_straight_lines(image):
    linesp = cv2.HoughLinesP(image, 0.6, 7 * np.pi / 180, 45, None, 90, 5)
    if linesp is not None:
        for i in range(0, len(linesp)):
            l = linesp[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0, 0, 0), 30, cv2.LINE_AA)
    return image


i = 0

filepath = easygui.fileopenbox()

if filepath.endswith('p'):
    IMGS = pickle.load(open(filepath, "rb"))
else:
    IMGS = read_video(filepath)

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

    th3 = cv2.adaptiveThreshold(log_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1.1)

    th3 = remove_too_small(th3, 100)

    th3 = remove_too_straight_lines(th3)

    th3 = remove_too_small(th3, 100)

    result = colorFilter(img,th3)
    result = bike_filter(result)
    result = remove_too_small(result, 110)

    score = np.sum(result) / 255

    if score > 1000:
        crackIndexList.append(i)
        if i + 1 in crackIndexList:
            f.write(f"\n        Another subsequent crack found. Score: {score}, Index: {i}")
        else:
            f.write(
                f"\nCracks detected at {datetime.timedelta(seconds=(i * 30) // FRAMESKIP)}\n    Score: {score},    Index: {i}")
            cv2.imwrite(f'detected_cracks/image{i}.png', img)
            cv2.imwrite(f'detected_cracks/image{i}_TH3_{score}.png', result)
            #  cv2.imwrite(f'detected_cracks/{i}image_mask.png', mask)
            # cv2.imwrite(f'detected_cracks/image_blur{i}.png', image_blur)
            #  cv2.imwrite(f'detected_cracks/image_log{i}.png',log_image)
            # cv2.imwrite(f'detected_cracks/image_th{i}.png', th3)
            time.sleep(0.1)
        cracks += 1
    i += 1
f.close()
print(cracks)
