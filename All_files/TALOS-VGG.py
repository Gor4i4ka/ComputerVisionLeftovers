import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import Loader
import tensorflow
import talos
#Globals
FRAME = 224
RGB = 3
PATH = 'D://IMGBUF2/'
MULT = True
CLASS = 3
PARTS = 3
EP = 3
BATCH = 900
NUM = 32
np.random.seed(1000)
PARAMETERS = {
    'batch_size': [1800],
    'epochs': [3, 4,10,20],
    'optimizer': ['Adam', 'SGD']
}
#Data Loading for VGG
X, Y = Loader.data_load(CLASS,PARTS,PATH, multi=True)

#Using colour histograms if needed
X = Loader.histogram(X, NUM, net=True)
FRAME = NUM

#X, TRAIN, TEST = Loader.preproc(X, reshape=False, normalize=False)
#X_train = np.copy(X[TRAIN])
#X_test = np.copy(X[TEST])
#del(X)

def model(x_train, y_train, x_val, y_val, params):

    #Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=64, input_shape=(FRAME,FRAME,3), kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 2nd convolutional layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 1st Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 3d convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 4th convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 2nd Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 5th convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 6th convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 7th convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 3d Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 8th convolutional layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 9th convolutional layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 10th convolutional layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 4d Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 11th convolutional layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 12th convolutional layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 13th convolutional layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 5d Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # Passing it to a Fully Commected layer
    model.add(Flatten())

    # 1st Fully Connected Layer
    #model.add(Dense(4096))
    model.add(Dense(4096, input_shape=(FRAME*FRAME*RGB,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    #model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    #model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    #model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(CLASS))
    model.add(Activation('softmax'))

    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=params['optimizer'], metrics=['accuracy'])
    # Training
    out = model.fit(x_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=[x_val, y_val])
    return out, model

talos.Scan(X, Y, params=PARAMETERS, model=model)
