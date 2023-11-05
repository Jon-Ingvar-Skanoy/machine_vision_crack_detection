import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import pickle
IMGS = pickle.load(open("images2.p", "rb"))

img = IMGS[260]
downscale = 1






imgblur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.pyrDown(imgblur)
imgblur = cv2.pyrDown(imgblur)

imgblur = cv2.GaussianBlur(imgblur, (5,5), 10000000000)
imgblur = cv2.medianBlur(imgblur,   3)






#imgblur = cv2.equalizeHist(imggray)
c = 255 / np.log(1 + np.max(imgblur))
log_image = c * (np.log(imgblur + 1))

# Specify the data type so that
# float value will be converted to int
log_image = np.array(log_image, dtype=np.uint8)
#log_image = cv2.GaussianBlur(log_image, (13,13), 40)
#bilateral = cv2.bilateralFilter(log_image, 50, 1, 1)
hist = cv2.calcHist([log_image],[0],None,[256],[0,256])
th3 = cv2.adaptiveThreshold(log_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,7,1.2)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th3, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:,cv2.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 300:   #keep
        result[labels == i + 1] = 255

#print(len(IMGS))
#bikefiltertestTH3 = copy.deepcopy(th3)
#bikefiltertest = copy.deepcopy(img)
#for i in range(int(300/downscale),int(1080/downscale)):
 #   for j in range(int(1200/downscale), int(2000/downscale)):
  #     bikefiltertestTH3[i][j] = 0
#th3 = bikefiltertestTH3
print(np.sum(th3)/255)
plt.hist(imgblur.ravel(),256,[0,256]); plt.show()

#plt.imshow(th3,  cmap='gray')
#plt.show()
cv2.imwrite('image.png', img)
#cv2.imwrite('image_gray.png', imggray)
cv2.imwrite('image_blur.png', imgblur)
cv2.imwrite('image_log.png',log_image)
cv2.imwrite('image_hist.png',hist)
cv2.imwrite('image_thresh.png', result)