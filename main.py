import cv2
import numpy as np

img = cv2.imread('image.jpg')

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "EDSR_x2.pb"
sr.readModel(path)
sr.setModel("edsr", 2)
 
result = sr.upsample(img)
 
# Resized image
resized = cv2.resize(img, dsize=None,fx=2,fy=2)

inBlack = np.array([40], dtype=np.float32)
inWhite = np.array([130], dtype=np.float32)
inGamma = np.array([1.3], dtype=np.float32)
outBlack = np.array([0], dtype=np.float32)
outWhite = np.array([255], dtype=np.float32)

img = np.clip( (img - inBlack) / (inWhite - inBlack), 0, 255 )
img = ( img ** (1/inGamma) ) * (outWhite - outBlack) + outBlack
img = np.clip(img, 0, 255).astype(np.uint8)

# show image in window fullscreen
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()