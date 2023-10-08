import numpy as np
import cv2

import matplotlib.pyplot as plt



img = cv2.imread('image.png')
img = cv2.pyrDown(img)
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray, (5,5), 1)

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

cv2.imwrite('image_gray.png', imggray)
cv2.imwrite('image_blur.png', imgblur)
cv2.imwrite('image_log.png',log_image)
cv2.imwrite('image_edge.png', edge)
cv2.imwrite('image_thresh.png', th3)