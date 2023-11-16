import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import pickle
IMGS = pickle.load(open("imagess.p", "rb"))

img = IMGS[301]
downscale = 4






imgblur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.pyrDown(imgblur)
imgblur = cv2.pyrDown(imgblur)

imgblur = cv2.GaussianBlur(imgblur, (5,5), 100000000000)
imgblur = cv2.medianBlur(imgblur,   9)


#imgblur = cv2.equalizeHist(imggray)
c = 255 / np.log(1 + np.max(imgblur))
log_image = c * (np.log(imgblur + 1))
log_image = np.array(log_image, dtype=np.uint8)
#log_image = cv2.GaussianBlur(log_image, (13,13), 40)
#bilateral = cv2.bilateralFilter(log_image, 50, 1, 1)
hist = cv2.calcHist([log_image],[0],None,[256],[0,256])
th3 = cv2.adaptiveThreshold(log_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,1)
print(np.sum(th3))


nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th3, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:,cv2.CC_STAT_AREA]

th3 = np.zeros((labels.shape), np.uint8)


for i in range(0, nlabels - 1):
    if areas[i] >= 70:   #keep
        th3[labels == i + 1] = 255

linesP = cv2.HoughLinesP(th3, 0.6, 7* np.pi / 180, 45, None, 90, 5)
if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(th3, (l[0], l[1]), (l[2], l[3]), (0,0,0), 30, cv2.LINE_AA)


greenfiltertestTH3 = copy.deepcopy(th3)
greenfiltertest = copy.deepcopy(img)
greenfiltertest = cv2.pyrDown(greenfiltertest)
greenfiltertest = cv2.pyrDown(greenfiltertest)

green_lower1 = (220, 220, 220)
green_upper1 = (255, 255, 255)
green_lower2 = (5, 56, 19)
green_upper2 = (69, 149, 100)

mask1 = cv2.inRange(greenfiltertest, green_lower1,green_upper1)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:,cv2.CC_STAT_AREA]

mask1 = np.zeros((labels.shape), np.uint8)


for i in range(0, nlabels - 1):
    if areas[i] >= 60:   #keep
        mask1[labels == i + 1] = 255

kernal = np.ones((2,2),np.uint8)
mask = cv2.bitwise_not(cv2.dilate(mask1,kernal,iterations=7))
result1 = cv2.bitwise_and(greenfiltertestTH3,mask)
mask2 = cv2.inRange(greenfiltertest, green_lower2,green_upper2)
result2 = cv2.bitwise_and(result1,cv2.bitwise_not(mask2))

bikefiltertestTH3 = copy.deepcopy(result1)
bikefiltertest = copy.deepcopy(result1)
for i in range(int(400/downscale),int(1080/downscale)):
    for j in range(int(1300/downscale), int(1900/downscale)):
       bikefiltertestTH3[i][j] = 0
result1 = bikefiltertestTH3

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(result1, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:,cv2.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)


for i in range(0, nlabels - 1):
    if areas[i] >= 100:   #keep
        result[labels == i + 1] = 255
print(len(IMGS))


print(np.sum(result)/255)
plt.hist(imgblur.ravel(),256,[0,256]); plt.show()





#plt.imshow(th3,  cmap='gray')
#plt.show()
cv2.imwrite('image.png', img)
cv2.imwrite('image_gray.png', result)
cv2.imwrite('image_blur.png', imgblur)
cv2.imwrite('image_log.png',log_image)
cv2.imwrite('image_hist.png',mask)
cv2.imwrite('image_thresh.png', result1)
cv2.imwrite('image_th3.png', th3)