import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import Loader
import tensorflow
import talos

# Существует для grid оптимизации talos'ом

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

#TALOS params
PARAMETERS0 = {
    'batch_size': [32, 64, 128, 256, 512, 900, 1575],
    'epochs': [3, 4, 10, 20],
    'optimizer': ['Adam', 'SGD'],
    'conv_layers': [3, 4, 5, 6, 7],
    'full_layers': [1, 2, 3],
    'filter_amount': [32, 64, 128],
    'filter_size': [0, 1, 2, 3],
    'full_size' : [512, 1024, 2048, 4096]
}
PARAMETERS = {
    'batch_size': [32, 64, 128, 256, 512],
    'epochs': [3, 4, 10, 20],
    'optimizer': ['Adam', 'SGD'],
    'conv_layers': [4, 5, 6],
    'full_layers': [1, 2],
    'filter_amount': [32, 64],
    'filter_size': [0, 1, 2, 3],
    'full_size' : [512, 1024]
}
PARAMETERS1 = {
    'batch_size': [256],
    'epochs': [40],
    'optimizer': ['Adam'],
    'conv_layers': [6],
    'full_layers': [1],
    'filter_amount': [32],
    'filter_size': [0],
    'full_size' : [1024]
}
PARAMETERSF = {
    'batch_size': [128, 256],
    'epochs': [15],
    'optimizer': ['Adam'],
    'conv_layers': [6, 7, 8],
    'full_layers': [1, 2],
    'filter_amount': [32, 64],
    'filter_size': [0, 1, 2, 3],
    'full_size' : [512, 768, 1024]
}

PARAMETERSF1 = {
    'batch_size': [128, 256],
    'epochs': [10],
    'optimizer': ['Adam'],
    'conv_layers': [6, 7, 8],
    'full_layers': [1, 2],
    'filter_amount': [64],
    'filter_size': [0, 1, 2, 3],
    'full_size' : [512, 768, 1024]
}
    #'batch_size': [128],
    #'epochs': [3],
    #'optimizer': ['SGD'],
    #'conv_layers': [6],
    #'full_layers': [2],
    #'filter_amount': [64],
    #'filter_size': [1],
    #'full_size' : [1024]

    #batch size  1800
    #epochs  4
    #optimizer  SGD
    #conv_layers  5
    #full_layers  1
    #filter_amount  128
    #filter_size  0
    #full_size  1024

def model(x_train, y_train, x_val, y_val, params):
    print(
        'batch size ', params['batch_size'], '\n'
        'epochs ', params['epochs'], '\n'
        'optimizer ', params['optimizer'], '\n'
        'conv_layers ', params['conv_layers'], '\n'
        'full_layers ', params['full_layers'], '\n'
        'filter_amount ', params['filter_amount'], '\n'
        'filter_size ', params['filter_size'], '\n'
        'full_size ', params['full_size'], '\n'
    )
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

    # Training
    out = model.fit(x_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=[x_val, y_val])
    return out, model

#Data Loading for Alexnet
X, Y = Loader.data_load(CLASS,PARTS,PATH, multi=True)
talos.Scan(X,Y, params=PARAMETERS, model= model)



