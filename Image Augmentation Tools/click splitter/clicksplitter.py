import cv2
import numpy as np


#VERTICAL MASK
V_IMAGE_HEIGHT=976
V_IMAGE_WIDTH= 976

#HORIZONTAL MASK

H_IMAGE_HEIGHT=976
H_IMAGE_WIDTH= 976

scan_type='v'
co_xy=[]
co_xy=np.array(co_xy,dtype='int')
co_xy=np.reshape(co_xy,(-1,2))

count=0

# mouse callback function
def note_xy(event,x,y,flags,param):
    global co_xy
    if event == cv2.EVENT_LBUTTONDOWN:
        print("last coordinate:",x,",",y)
        xy=np.array([[x,y]])
        co_xy=np.append(co_xy,xy,axis=0)
        print(co_xy)
        if(scan_type=='v'):
            draw_v(x,y)
            save_and_draw_slice_v(x,y)
        else:
            draw_h(x,y)
            save_and_draw_slice_h(x,y)

def draw_v(x,y):
    cv2.circle(img,(x,y),3,(0,255,255),-1)
    
def draw_h(x,y):
    cv2.circle(img,(x,y),3,(0,0,255),-1)

def save_and_draw_slice_v(x,y):
    global count
    x_left=x-(V_IMAGE_WIDTH/2)
    y_left=y-(V_IMAGE_HEIGHT/2)
    print("upper corner:",x_left,",",y_left)
    img_slice=img_for_save[int(y_left):int(y_left+V_IMAGE_HEIGHT),int(x_left):int(x_left+V_IMAGE_WIDTH)]
    cv2.imwrite(str(count)+"vertical_slice.png",img_slice)
    count+=1
    

def save_and_draw_slice_h(x,y):
    global count
    x_left=x-(H_IMAGE_WIDTH/2)
    y_left=y-(H_IMAGE_HEIGHT/2)
    print("upper corner:",x_left,",",y_left)
    img_slice=img_for_save[int(y_left):int(y_left+H_IMAGE_HEIGHT),int(x_left):int(x_left+H_IMAGE_WIDTH)]
    cv2.imwrite(str(count)+"horizontal_slice.png",img_slice)
    count+=1
    
img_for_save = cv2.imread('2.png',1)
img = cv2.imread('2.png',1)
cv2.rectangle(img,(int(V_IMAGE_WIDTH/2),int(V_IMAGE_HEIGHT/2)),(img.shape[1]-int(V_IMAGE_WIDTH/2),img.shape[0]-int(V_IMAGE_HEIGHT/2)),(0,255,255),2)
cv2.rectangle(img,(int(H_IMAGE_WIDTH/2),int(H_IMAGE_HEIGHT/2)),(img.shape[1]-int(H_IMAGE_WIDTH/2),img.shape[0]-int(H_IMAGE_HEIGHT/2)),(0,0,255),2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',note_xy)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 104:
        scan_type='h' 
    if cv2.waitKey(20) & 0xFF == 118:
        scan_type='v'
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()