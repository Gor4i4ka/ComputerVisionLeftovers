import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, LeakyReLU, \
                        MaxPooling2D, Input, Concatenate, Layer, Lambda, UpSampling2D, BatchNormalization
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import numpy as np
import Loader
import math
import tensorflow as tf
from keras.layers import Layer
import talos
from PIL import Image
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils

def resize_bilinear(x, RESIZE_FACTOR=2):
    return tf.image.resize_bilinear(x, size=[K.shape(x)[1]*RESIZE_FACTOR, K.shape(x)[2]*RESIZE_FACTOR])

# constants
FRAME = 384
RGB = 3
PATH = 'D://IMGBUF1/'
PATH_TEST = 'D://'
MULT = True
CLASS = 4
PARTS = 1
EP = 3
BATCH = 128
NUM = 32
np.random.seed(1000)
BLOCK_AMOUNT = 32
T_FIX = 90.0
# load the images

#X, Y = Loader.data_load(CLASS, PARTS, PATH, multi=True)

# An image
#img0 = X[2]
#img0 = X[275]
#img0 = X[512]
#img0 = X[850]
#IMG0 = Image.fromarray(img0.astype('uint8'), 'RGB')
#IMG0.show()

#Grayification
def rgb2gray(img, r=0.297, g=0.588, b=0.115):
    return r * img[:, :, 0] + g * img[:, :, 1] + b * img[:, :, 2]
    #IMG1 = Image.fromarray(img1.astype('uint8'), 'L')
    #IMG1.show()

#Removal of background
def bg_remove(img, BLOCK_AMOUNT=32, FRAME=384, T_FIX=90.0):
    rest = FRAME % BLOCK_AMOUNT
    normal_block_size = math.floor(FRAME / BLOCK_AMOUNT)
    last_block_size = normal_block_size + rest
    img_buf = np.copy(img)

    for block in range(BLOCK_AMOUNT * BLOCK_AMOUNT):
        i = math.floor(block / BLOCK_AMOUNT)
        j = block % BLOCK_AMOUNT
        if (i < BLOCK_AMOUNT - 1) and (j < BLOCK_AMOUNT - 1):
            extracted_block = np.copy(img[i * normal_block_size : i * normal_block_size + normal_block_size,
                                    j * normal_block_size : j * normal_block_size + normal_block_size])
        else:
            extracted_block = np.copy(img[i * normal_block_size : i * normal_block_size + last_block_size,
                                    j * normal_block_size : j * normal_block_size + last_block_size])
        block_min = extracted_block.min()
        block_max = extracted_block.max()
        block_intensity_variance = block_max - block_min
        #block_img = Image.fromarray(extracted_block.astype('uint8'), 'L')
        #IMG2.show()

        if block_intensity_variance < T_FIX:
            img_buf[i * normal_block_size : i * normal_block_size + extracted_block.shape[0],
                    j * normal_block_size : j * normal_block_size + extracted_block.shape[0]] = 0
    return img_buf
    #IMG2 = Image.fromarray(img_buf.astype('uint8'), 'L')
    #IMG2.show()

def test_preproc(data):
    for img_ind in range(data.shape[0]):
        print(img_ind)
        IMG = Image.fromarray(data[img_ind].astype('uint8'), 'RGB')
        IMG.save('D://IMGINIT/' + str(img_ind) + '.PNG', 'PNG')
        img1 = rgb2gray(np.copy(data[img_ind]))
        img2 = bg_remove(img1)
        IMG1 = Image.fromarray(img2.astype('uint8'), 'L')
        IMG1.save('D://IMGPREPROC/' + str(img_ind) + '.PNG', 'PNG')

#test_preproc(X)

def east_loss(y_true, y_pred):
    sc_true = y_true[0]
    sc_pred = y_pred[0]
    #bb_true = y_true[1]
    #bb_pred = y_pred[1]

    #print(y_true)

    B = 1 - K.mean(sc_true)

    cl_loss = (-1) * B * sc_true * K.log(sc_pred + 1e-6) - (1 - B) * (1 - sc_true) * K.log(1 - sc_pred + 1e-6)
    res = K.sum(cl_loss)

    return res


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def east_net():

    #PVA NET

    main_input = Input(shape=(512, 512, 3), name='main_input')

    conv0 = Conv2D( filters=16,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    padding='same',
                    name='conv0')(main_input)
    conv0 = LeakyReLU(alpha=0.1)(conv0)
    #conv0 = BatchNormalization()(conv0)

    conv1 = Conv2D( filters=64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='same',
                    name='conv1')(conv0)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    #conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(filters=128,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   name='conv2')(conv1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    #conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   name='conv3')(conv2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    #conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(filters=384,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   name='conv4')(conv3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    #conv4 = BatchNormalization()(conv4)
    # PVA_NET END
    # FEATURE MERGING BRANCH

    #unpool1 = MaxUnpooling2D(size=(2, 2))(conv4)
    #unpool1 = Lambda(resize_bilinear, name='resize1')(conv4)
    unpool1 = UpSampling2D(size=(2, 2))(conv4)

    concat1 = Concatenate(axis=-1)([unpool1, conv3])

    convM1_1 = Conv2D(filters=128,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      name='convM1_1')(concat1)
    convM1_1 = LeakyReLU(alpha=0.1)(convM1_1)
    #convM1_1 = BatchNormalization()(convM1_1)

    convM1_2 = Conv2D(filters=128,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      name='convM1_2')(convM1_1)
    convM1_2 = LeakyReLU(alpha=0.1)(convM1_2)
    #convM1_2 = BatchNormalization()(convM1_2)

    #unpool2 = MaxUnpooling2D(size=(2, 2))(convM1_2)
    unpool2 = UpSampling2D(size=(2, 2)) (convM1_2)

    concat2 = Concatenate(axis=-1)([unpool2, conv2])

    convM2_1 = Conv2D(filters=64,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      name='convM2_1')(concat2)
    convM2_1 = LeakyReLU(alpha=0.1)(convM2_1)
    #convM2_1 = BatchNormalization()(convM2_1)

    convM2_2 = Conv2D(filters=64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      name='convM2_2')(convM2_1)
    convM2_2 = LeakyReLU(alpha=0.1)(convM2_2)
    #convM2_2 = BatchNormalization()(convM2_2)

    #unpool3 =  MaxUnpooling2D(size=(2, 2))(convM2_2)
    unpool3 = UpSampling2D(size=(2, 2))(convM2_2)

    concat3 = Concatenate(axis=-1)([unpool3, conv1])

    convM3_1 = Conv2D(filters=32,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      name='convM3_1')(concat3)
    convM3_1 = LeakyReLU(alpha=0.1)(convM3_1)
    #convM3_1 = BatchNormalization()(convM3_1)

    convM3_2 = Conv2D(filters=32,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      name='convM3_2')(convM3_1)
    convM3_2 = LeakyReLU(alpha=0.1)(convM3_2)
    #convM3_2 = BatchNormalization()(convM3_2)

    convM4 = Conv2D(filters=32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    name='convM4')(convM3_2)
    convM4 = LeakyReLU(alpha=0.1)(convM4)
    #convM4 = BatchNormalization()(convM4)

    score = Conv2D( filters=1,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='same',
                    name='conv_score')(convM4)
    score = LeakyReLU(alpha=0.1)(score)

    score = Flatten()(score)


    bbox = Conv2D(  filters=4,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='same',
                    name='conv_bbox')(convM3_2)
    bbox = LeakyReLU(alpha=0.1)(bbox)


    model = Model(inputs=main_input, outputs=score)
    #model = Model(inputs=[main_input], outputs=score)

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    dt = np.load("D://DATACOCO/DATA/data_part1.npy")
    sc = np.load("D://DATACOCO/SCORE/score_part1.npy")
    #sc = sc.flatten()
    bb = np.load("D://DATACOCO/BOUND/bound_part1.npy")
    print(1 - np.mean(sc[0]))

    #print(sc.shape)
    #for i in range(3):
        #dat = dt[i]
        #scr = sc[i, :, :, 0]
        #IMG = Image.fromarray(dat.astype('uint8'), 'RGB')
        #IMG.show()
        #IMG = Image.fromarray(scr.astype('uint8'), 'L')
       # IMG.show()
    learn = 9
    img = 1
    dt = (dt - dt.mean(axis=0)) / dt.max(axis=0)
    cur_batch = 1
    batch_am = 2
    for i in range(10):
        #dt = np.load("D://DATACOCO/DATA/data_part" + str(cur_batch) + ".npy")
        #sc = np.load("D://DATACOCO/SCORE/score_part" + str(cur_batch) + ".npy")
        #sc = sc.astype('int')
        print("YOHOHOH", cur_batch)
        print(model.train_on_batch(x=dt, y=sc))
        cur_batch += 1
        if cur_batch > batch_am:
            cur_batch = 1
    res = (model.predict(dt)) * 200
    IMG = Image.fromarray(res[img, :, :, 0].astype('uint8'), 'L')
    IMG.show()
    #print('KEK')

    #model.train_on_batch
east_net()
