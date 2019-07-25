#this code uses U-NET FROM LBP(SINGLE CHANNEL) TO PRODUCE RGB OUTPUT

import numpy as np
import keras
import os
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.callbacks import EarlyStopping




#####################################################

X_train=[]

for i in os.listdir('train lbp rgb pure/inputs'):
    if '.png' in i:
        X_train=np.append(X_train,i)

Y_train=X_train     
        
X_valid=[]

for i in os.listdir('valid lbp rgb pure/inputs'):
    if '.png' in i:
        X_valid=np.append(X_valid,i)

Y_valid=X_valid

########################################################
        
images={ 'train':X_train, 
             'validation':X_valid}

masks={ 'train':Y_train, 
        'validation':Y_valid}

#########################################################

print(images['validation'],masks['validation'])
print(images['train'],masks['train'])

#########################################################

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, n_channels,path_X,path_Y,shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path_X=path_X
        self.path_Y=path_Y
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        
        X=X/255.0
        Y=Y/255.0
        #print("Batch was fed to GPU!")
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 3))
    
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,:,:,0] = cv2.imread(self.path_X + ID,0)

            # Store class
            Y[i,:,:,:] = cv2.imread(self.path_Y + ID,1)

        return X, Y
    
#########################################################
    
path_to_X_train='train lbp rgb pure/inputs/'
path_to_Y_train='train lbp rgb pure/masks/'

path_to_X_valid='valid lbp rgb pure/inputs/'
path_to_Y_valid='valid lbp rgb pure/masks/'

batch_size_train=3
batch_size_valid=2


    
training_generator = DataGenerator(list_IDs=images['train'], labels=masks['train'], \
                                   batch_size=batch_size_train, dim=(976,976),\
                                   n_channels=1, path_X=path_to_X_train,path_Y=path_to_Y_train, \
                                   shuffle=True)

validation_generator = DataGenerator(list_IDs=images['validation'], labels=masks['validation'],\
                                     batch_size=batch_size_valid, dim=(976,976),\
                                   n_channels=1, path_X=path_to_X_valid, path_Y=path_to_Y_valid, \
                                   shuffle=True)

#########################################################


#note the indexes can be passed till __len__ only that is (0 to (__len__-1))
#!since floor is used some image are always left ie remainder when divided by batch size! :/

es = EarlyStopping(monitor='val_dice_coef', min_delta=0.001,mode='max', verbose=1,patience=3,restore_best_weights=True)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

#########################################################

# Build U-Net model
inputs = Input((976, 976, 1))

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)
#ouput 248

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)
#ouput 124

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)
#ouput 62

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
#ouput 31


c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
#ouput 62

u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
#ouput 124
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
#ouput 248
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
#ouput 496
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(3, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
model.summary()


model.fit_generator(generator=training_generator,validation_data=validation_generator, \
                    epochs=2, verbose=2, callbacks=[es])

model.save('3channel.h5')





