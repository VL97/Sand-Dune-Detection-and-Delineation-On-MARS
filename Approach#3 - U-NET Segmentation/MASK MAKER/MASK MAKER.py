import cv2
import numpy as np

img=cv2.imread("DRAWN.png",1)


print(img.shape)

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if(np.all(img[x,y]==[36,28,237])):
            img[x,y]=[255,255,255]
            print("WORKING!")
        else:
            img[x,y]=[0,0,0]
            
            
            
cv2.imwrite("MASK_BOUNDARY.png",img)

