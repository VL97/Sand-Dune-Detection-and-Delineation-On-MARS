import cv2
import os

file_images=[]

for x in os.listdir("to flip"):
    x='to flip/'+x
    if '.png' in x:	
    	file_images=np.append(file_images,x)
    
print(file_images)

for i in range(len(file_images)):
    image=cv2.imread(file_images[i],1)
    #rotate 180 degree
    image1=cv2.flip(image,-1)
    #flip horizontally
    image2=cv2.flip(image,1)
    #flipped first horizontally and then rotate 180 degree OR flipped vertically
    image3=cv2.flip(image,0)
    cv2.imwrite('flipped/'+str(i)+'1.png',image1)
    cv2.imwrite('flipped/'+str(i)+'2.png',image2)
    cv2.imwrite('flipped/'+str(i)+'3.png',image3)
    print("working!")
    
    
