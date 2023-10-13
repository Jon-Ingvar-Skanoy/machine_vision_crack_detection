import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import pickle
IMGS = pickle.load(open("imagess.p", "rb"))

img = IMGS[300]
downscale = 1
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#imggray = cv2.pyrDown(imggray)
#imggray = cv2.pyrDown(imggray)

imgblur = cv2.GaussianBlur(imggray, (13,13), 40)
imgblur = cv2.medianBlur(imgblur,   15)


#imgblur = cv2.equalizeHist(imggray)
c = 255 / np.log(1 + np.max(imgblur))
log_image = c * (np.log(imgblur + 1))

# Specify the data type so that
# float value will be converted to int
log_image = np.array(log_image, dtype=np.uint8)

bilateral = cv2.bilateralFilter(log_image, 50, 1, 1)
hist = cv2.calcHist([bilateral],[0],None,[256],[0,256])
th3 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,1)
th3 = cv2.medianBlur(th3,   7)

print(len(IMGS))
bikefiltertestTH3 = copy.deepcopy(th3)
bikefiltertest = copy.deepcopy(img)
for i in range(int(440/downscale),int(1080/downscale)):
    for j in range(int(1200/downscale), int(1940/downscale)):
       bikefiltertestTH3[i][j] = 0
th3 = bikefiltertestTH3
print(bilateral.max())
plt.hist(imgblur.ravel(),256,[0,256]); plt.show()

#plt.imshow(th3,  cmap='gray')
#plt.show()
cv2.imwrite('image.png', img)
cv2.imwrite('image_gray.png', imggray)
cv2.imwrite('image_blur.png', imgblur)
cv2.imwrite('image_log.png',log_image)
cv2.imwrite('image_hist.png',hist)
cv2.imwrite('image_thresh.png', th3)