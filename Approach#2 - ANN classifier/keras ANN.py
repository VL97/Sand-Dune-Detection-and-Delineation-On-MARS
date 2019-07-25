import skimage.feature
import os
import numpy as np
import keras 
from keras import backend as K
from keras.models import Sequential
from keras.layers  import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import tensorflow as tf

radius=6
n_points=15
METHOD="uniform"

#TEXTURE1 files
sand_ripple=[]

for x in os.listdir("sand_ripple"):
    x='sand_ripple/'+x
    if '.png' in x:
        sand_ripple=np.append(sand_ripple,x)
    
print(sand_ripple)

#TEXTURE2 files
sand_fine=[]

for x in os.listdir("sand_fine"):
    x='sand_fine/'+x
    if '.png' in x:
        sand_fine=np.append(sand_fine,x)
    
print(sand_fine)

#TEXTURE3 files
terrain=[]

for x in os.listdir("terrain"):
    x='terrain/'+x
    if '.png' in x:
        terrain=np.append(terrain,x)
    
print(terrain)

#all texture files
texture=[]
texture=np.append(texture,sand_ripple)
texture=np.append(texture,sand_fine)
texture=np.append(texture,terrain)

#defined FEATURES
X=[]


#generating all features and appending to X
for i in range(len(texture)):
    image= skimage.io.imread(texture[i],as_grey=True)
    lbp = skimage.feature.local_binary_pattern(image, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    a=np.histogram(lbp.ravel(),normed=True, bins=n_bins, range=(0, n_bins))
    X=np.append(X,a[0])

#generating feature table/matrix    
X=np.reshape(X,(len(texture),n_bins))

#LABEL FOR TEXTURE1
y1=np.ones(len(sand_ripple),int)                         #MUST BE single array not matrix
#LABEL FOR TEXTURE2
y2=np.ones(len(sand_fine),int)                            #MUST BE single array not matrix
#LABEL FOR TEXTURE2
y3=np.zeros(len(terrain),int)                            #MUST BE single array not matrix

#ALL LABELS
y=[]
y=np.append(y,y1)
y=np.append(y,y2)
y=np.append(y,y3) 
                                     
print(X)
print(y)
################TEST DATA########

test_images=['test/test (1).png',
             'test/test (2).png',
             'test/test (3).png',
             'test/test (4).png',
             'test/test (5).png',
             'test/test (6).png',
             'test/test (7).png',
             'test/test (8).png',
             'test/test (9).png',
             'test/test (10).png',
             'test/test (11).png',
             'test/test (12).png',
             'test/test (13).png',
             'test/test (14).png',
             'test/test (15).png',
             'test/test (16).png']

label_test=[0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,0]
test_X=[] 

for i in range(len(test_images)):
    image= skimage.io.imread(test_images[i],as_grey=True)
    lbp = skimage.feature.local_binary_pattern(image, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    a=np.histogram(lbp.ravel(),normed=True, bins=n_bins, range=(0, n_bins))
    test_X=np.append(test_X,a[0])
    

test_X=np.reshape(test_X,(len(test_images),n_bins))
print(test_X)

##################### KERAS ##########
model=Sequential([
        Dense(16,input_shape=(17,),activation='relu'),
        Dense(32,activation='relu'),
        Dense(2,activation='softmax')
        ])

model.summary()

model.compile(Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#takes last only for validation_split :/
model.fit(X,y,batch_size=10,epochs=10,shuffle=True,verbose=2)

predict=model.predict(test_X,verbose=0)

rounded_prediction=model.predict_classes(test_X,verbose=0)
result=[]

for i in range(test_X.shape[0]):
    print(predict[i]," $ ",rounded_prediction[i]," $ ",label_test[i])
    if(rounded_prediction[i]==label_test[i]):
        result=np.append(result,1)
    else:
        result=np.append(result,0)

print(result)
print(np.mean(result))
