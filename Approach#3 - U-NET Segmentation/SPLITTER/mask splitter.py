import cv2
import  numpy as np
import matplotlib.pyplot
import skimage.feature
from skimage.filters import roberts, sobel, scharr, prewitt

'''
#only for testing purposes!
def image_draw(y_start,y_end,x_start,x_end):
    global image_for_output
    global count
    #draws a rectangle on the image from specified points
    if(count%2==0):
        width=3    
    else:
        width=6
    image_for_output = cv2.rectangle(image_for_output,(x_start,y_start),(x_end,y_end),(0,255,0),width)
    count+=1
'''

#the greyscale image to be analyzed
image=cv2.imread("FINAL_MASK.png",0)
#image_for_output=cv2.imread("LBP.png",1)


vertical_mask_length=976
vertical_mask_height=976
pass_density=2

#HORIZONTAL
number_of_whole_passes_possible_x=int(image.shape[1]/vertical_mask_length)-1          #we will leave the last pass as its traversed in the 2nd last 
density_passes_made_x=pass_density*number_of_whole_passes_possible_x
x_spacing=int(vertical_mask_length/pass_density)

#VERTICAL
number_of_whole_passes_possible_y=int(image.shape[0]/vertical_mask_height)-1        #we will leave the last pass as its traversed in the 2nd last 
density_passes_made_y=pass_density*number_of_whole_passes_possible_y
y_spacing=int(vertical_mask_height/pass_density)

rows = 976
cols = 976
count=0

for i in range(density_passes_made_x):
    x_start=i*x_spacing
    for j in range(density_passes_made_y):
        y_start=j*y_spacing
        image_sliced=image[y_start: (y_start+vertical_mask_height), x_start : (x_start + vertical_mask_length)]      #OPPOSITE FORMAT!!    (HEIGHT!!!=ROWS=X)
        #image_draw(y_start,y_start+vertical_mask_height,x_start,x_start + vertical_mask_length)
               
        flipped1 = cv2.flip(image_sliced, 1)
        flipped0 = cv2.flip(image_sliced, 0)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dst = cv2.warpAffine(image_sliced,M,(cols,rows))
   
        cv2.imwrite("MASKS FOR TRAINING/"+str(i)+str(j)+"a_orig.png",image_sliced)
        cv2.imwrite("MASKS FOR TRAINING/"+str(i)+str(j)+"b_rotated.png",dst)
        cv2.imwrite("MASKS FOR TRAINING/"+str(i)+str(j)+"c_horiflip.png",flipped1)
        cv2.imwrite("MASKS FOR TRAINING/"+str(i)+str(j)+"d_vertiflip.png",flipped0)
        
        print("making slices!")
        
#cv2.imwrite('TEMP.PNG',image_for_output)

