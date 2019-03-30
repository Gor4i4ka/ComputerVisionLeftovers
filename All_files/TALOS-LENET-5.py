from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import Loader
import talos
#Globals
FRAME = 224
RGB = 3
PATH = 'D://IMGBUF2/'
MULT = True
CLASS = 3
PARTS = 3
EP = 10
BATCH = 128
NUM = 32
np.random.seed(1000)
PARAMETERS = {
    'batch_size': [32,64,128,256,512,900,1800],
    'epochs': [3, 4,10,20],
    'optimizer': ['Adam', 'SGD']
}

#Data Loading for LENET
X, Y = Loader.data_load(CLASS,PARTS,PATH, multi=True)

#Using colour histograms if needed
X = Loader.histogram(X, NUM, net=True)
FRAME = NUM

#X, TRAIN, TEST = Loader.preproc(X, reshape=False, normalize=False)
#X_train = np.copy(X[TRAIN])
#X_test = np.copy(X[TEST])
#del(X)
#Instantiate an empty model
def model(x_train, y_train, x_val, y_val, params):
    print('ALOHA ', params['batch_size'], '\n',
          'BOB ', params['epochs'], '\n')
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(FRAME,FRAME,RGB)))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))
    #output layer
    model.add(Dense(units=CLASS, activation = 'softmax'))
    model.summary()

    # Compile the model
    model.compile(loss=losses.categorical_crossentropy, optimizer=params['optimizer'], metrics=['accuracy'])

    #Training
    out = model.fit(x_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=[x_val, y_val])
    return out, model
talos.Scan(X,Y, params=PARAMETERS, model= model)
