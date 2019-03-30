import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import Loader
import tensorflow
import talos

# Существует для проверки конкретной grid оптимизированной сети talos'ом

# constants
FRAME = 224
RGB = 3
PATH = 'D://IMGBUF2/'
MULT = True
CLASS = 3
PARTS = 3
EP = 3
BATCH = 128
NUM = 32
np.random.seed(1000)

params = {
    'batch_size': 128,
    'epochs': 100,
    'optimizer': 'Adam',
    'conv_layers': 7,
    'full_layers': 2,
    'filter_amount': 32,
    'filter_size': 0,
    'full_size' : 768
}

#Data Loading for Custom
X, Y = Loader.data_load(CLASS,PARTS,PATH, multi=True)

X, TRAIN, TEST = Loader.preproc(X, reshape=False, normalize=False)
X_train = np.copy(X[TRAIN])
X_test = np.copy(X[TEST])

print(X_train.shape)

del(X)

# instantiate an empty model
model = Sequential()

for conv_layer in range(params['conv_layers']):

    if conv_layer == 0:
        if params['filter_size'] == 1 or params['filter_size'] == 3:
            print('CONV0_1 ')
            model.add(Conv2D(filters=params['filter_amount'], input_shape=(FRAME, FRAME, RGB),
                                 kernel_size=(11, 11),
                                 strides=(2, 2), padding='valid'))
            model.add(Activation('relu'))
        else:
            print('CONV0_2 ')
            model.add(Conv2D(filters=params['filter_amount'], input_shape=(FRAME, FRAME, RGB),
                             kernel_size=(3, 3),
                                 strides=(2, 2), padding='valid'))
            model.add(Activation('relu'))
    if conv_layer == 1:
        if params['filter_size'] == 2 or params['filter_size'] == 3:
            print('CONV1_1 ')
            model.add(Conv2D(filters=params['filter_amount'],
                                kernel_size=(11, 11),
                                strides=(2, 2), padding='valid'))
            model.add(Activation('relu'))
        else:
            print('CONV1_2 ')
            model.add(Conv2D(filters=params['filter_amount'],
                                kernel_size=(3, 3),
                                strides=(2, 2), padding='valid'))
            model.add(Activation('relu'))
    if conv_layer > 1:
        print('CONV2_1 ')
        model.add(Conv2D(filters=params['filter_amount'],
                            kernel_size=(3, 3),
                            strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

    if conv_layer != 0 and conv_layer % 2 == 0 and conv_layer < 6:
        if params['conv_layers'] <= 6:
            print('POOL1_0 ')
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid'))
        else:
            print('POOL1_2')
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

print('FLAT ')
model.add(Flatten())

for full_layer in range(params['full_layers']):
    print('FULL ')
    model.add(Dense(params['full_size']))
    model.add(Activation('relu'))

print('CLASS ')
model.add(Dense(CLASS))
model.add(Activation('softmax'))

# model summary
model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=params['optimizer'], metrics=['accuracy'])

#Training
model.fit(X_train, np.copy(Y[TRAIN]), epochs=params['epochs'], batch_size=params['batch_size'])
score = model.evaluate(X_test, np.copy(Y[TEST]), batch_size=params['batch_size'])
print("SCORE IS ", score)


