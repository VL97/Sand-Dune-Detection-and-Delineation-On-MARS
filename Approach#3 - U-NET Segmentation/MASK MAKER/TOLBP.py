import cv2
import os
import skimage.feature
import numpy as np



img=cv2.imread('INPUT.png',0)    
lbp = skimage.feature.local_binary_pattern(image=img, P=1, R=20)
'''
cv2.imshow("",lbp)
cv2.waitKey()
cv2.destroyAllWindows()
'''
cv2.imwrite('LBP.png',lbp*255)
