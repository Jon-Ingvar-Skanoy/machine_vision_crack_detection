import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import pickle
IMGS = pickle.load(open("imagesS.p", "rb"))

img = IMGS[221]
downscale = 1






imgblur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.pyrDown(imgblur)
imgblur = cv2.pyrDown(imgblur)

imgblur = cv2.GaussianBlur(imgblur, (5,5), 100000000000)
imgblur = cv2.medianBlur(imgblur,   9)






#imgblur = cv2.equalizeHist(imggray)
c = 255 / np.log(1 + np.max(imgblur))
log_image = c * (np.log(imgblur + 1))

# Specify the data type so that
# float value will be converted to int
log_image = np.array(log_image, dtype=np.uint8)
#log_image = cv2.GaussianBlur(log_image, (13,13), 40)
#bilateral = cv2.bilateralFilter(log_image, 50, 1, 1)
hist = cv2.calcHist([log_image],[0],None,[256],[0,256])
th3 = cv2.adaptiveThreshold(log_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,7,1)



nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th3, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:,cv2.CC_STAT_AREA]

th3 = np.zeros((labels.shape), np.uint8)


for i in range(0, nlabels - 1):
    if areas[i] >= 100:   #keep
        th3[labels == i + 1] = 255

linesP = cv2.HoughLinesP(th3, 0.6, 10* np.pi / 180, 45, None, 90, 5)
if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(th3, (l[0], l[1]), (l[2], l[3]), (255,0,0), 30, cv2.LINE_AA)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th3, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:,cv2.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)


for i in range(0, nlabels - 1):
    if areas[i] >= 100:   #keep
        result[labels == i + 1] = 255
print(len(IMGS))
#bikefiltertestTH3 = copy.deepcopy(th3)
#bikefiltertest = copy.deepcopy(img)
#for i in range(int(300/downscale),int(1080/downscale)):
 #   for j in range(int(1200/downscale), int(2000/downscale)):
  #     bikefiltertestTH3[i][j] = 0
#th3 = bikefiltertestTH3

print(np.sum(result)/255)
plt.hist(imgblur.ravel(),256,[0,256]); plt.show()

greenfiltertestTH3 = copy.deepcopy(result)
greenfiltertest = copy.deepcopy(img)
greenfiltertest = cv2.pyrDown(greenfiltertest)
greenfiltertest = cv2.pyrDown(greenfiltertest)

green_lower1 = (20,71,34)
green_upper1 = (121,193,154)
green_lower2 = (5,56,19)
green_upper2 = (69,149,100)

mask1 = cv2.inRange(greenfiltertest, green_lower1,green_upper1)
result1 = cv2.bitwise_and(greenfiltertestTH3,cv2.bitwise_not(mask1))
mask2 = cv2.inRange(greenfiltertest, green_lower2,green_upper2)
result2 = cv2.bitwise_and(result1,cv2.bitwise_not(mask2))
cv2.imshow("mask2", cv2.bitwise_not(mask2))
cv2.waitKey(0)

#plt.imshow(th3,  cmap='gray')
#plt.show()
cv2.imwrite('image.png', img)
#cv2.imwrite('image_gray.png', imggray)
cv2.imwrite('image_blur.png', imgblur)
cv2.imwrite('image_log.png',log_image)
cv2.imwrite('image_hist.png',hist)
cv2.imwrite('image_thresh.png', result)