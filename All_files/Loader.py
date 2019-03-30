#Libraries
#from matplotlib import pyplot as py
from PIL import Image
import numpy as np
import glob
import os
import scipy.misc as rs
import pandas as pd
from sklearn.model_selection import train_test_split as sk1
from math import floor
from sklearn import preprocessing
from random import randint

# Global Vars
SIZE = 250
#FRAME = 384
FRAME = 224
RGB = 3
#PATH = 'D://IMGBUF1/'
def polynom(X, GRADE):
    X_R = np.zeros((X.shape[0], GRADE*X.shape[1]))
    for i in range(X_R.shape[0]):
        for j in range(X_R.shape[1]):
            X_R[i][j] = X[i][j%X.shape[1]]
            BASE = X_R[i][j]
            for k in range(floor(j/X.shape[1])):
                X_R[i][j] *= BASE
    #print(X_R)
    return X_R
#Economic norm
def norm(X):
    print(X.shape)
    for i in range(X.shape[0]):
        #print(i)
        if X[i].max() != 0:
            X[i] = X[i]/X[i].max()

#histogram for tree and linear
def histogram(X, NUM, net=False):
    global RGB
    global FRAME

    if not net:
        X_H = np.zeros((X.shape[0], NUM, RGB))
        BINS = np.zeros((NUM + 1))
        for i in range(1, NUM + 1):
            BINS[i] = NUM * i
    else:
        X_H = np.zeros((X.shape[0], NUM, NUM, RGB))
        BINS = np.zeros((NUM * NUM + 1))
        for i in range(1, NUM * NUM + 1):
            BINS[i] = NUM * i

    for i in range(X.shape[0]):
        X_RED = X[i, :, :, 0].flatten()
        X_GRE = X[i, :, :, 1].flatten()
        X_BLU = X[i, :, :, 2].flatten()
        #print(X_RED.shape, X_GRE.shape, X_BLU.shape)
        #print(X_RED, X_GRE, X_BLU)
        X_RED = np.histogram(X_RED, bins=BINS)[0]
        X_GRE = np.histogram(X_GRE, bins=BINS)[0]
        X_BLU = np.histogram(X_BLU, bins=BINS)[0]
        #print(X_RED.shape, X_GRE.shape, X_BLU.shape)
        #print(X_RED, X_GRE, X_BLU)
        if not net:
            for j in range(NUM):
                X_H[i, :, 0] = X_RED
                X_H[i, :, 1] = X_GRE
                X_H[i, :, 2] = X_BLU
        else:
            X_RED = np.reshape(X_RED, ((NUM, NUM)))
            X_BLU = np.reshape(X_BLU, ((NUM, NUM)))
            X_GRE = np.reshape(X_GRE, ((NUM, NUM)))

            X_H[i, :, :, 0] = X_RED
            X_H[i, :, :, 1] = X_GRE
            X_H[i, :, :, 2] = X_BLU
    print('HISTOGRAM DONE:', X_H.shape)
    return X_H

#Loading npy arrays
def data_load(CLASSES, PARTS, PATH, multi=False):
    global SIZE
    if multi:
        Y = np.zeros((CLASSES*PARTS*SIZE, CLASSES))
    else:
        Y = np.zeros((CLASSES*PARTS*SIZE))
    X = np.zeros((CLASSES*PARTS*SIZE, FRAME, FRAME, RGB))
    # size of BUF: BUF_X = np.zeros((SIZE, FRAME, FRAME, RGB))
    IND = 0
    for prt in range(PARTS):
        for cls in range(CLASSES):
            LOC_PATH = PATH + 'data' + str(cls) + 'part' + str(prt) + '.npy'
            BUF_X = np.load(LOC_PATH)

            for i in range(BUF_X.shape[0]):
                X[IND] = BUF_X[i]
                if multi:
                    Y[IND][cls] = 1
                else:
                    Y[IND] = cls
                IND += 1
            print('LOADED ', LOC_PATH)
    return X, Y


def preproc(X, reshape=True, normalize=True):
    #RESHAPE
    if reshape:
        X = X.reshape((X.shape[0], int(np.prod(X.shape)/X.shape[0])))
    #NORMALIZE
    if normalize:
        X = X.T
        #X = preprocessing.normalize(X, norm='max') too much memory
        norm(X)
        X = X.T
    #Splitting the data
    SEED = randint(0,9)
    TEST_SIZE = 0.20
    INDECIES = np.arange(X.shape[0])
    TRAIN, TEST = sk1(INDECIES, test_size=TEST_SIZE, random_state=SEED)
    #print(TRAIN.shape, TEST.shape, SEED)
    print("PREPROC DONE")
    return X, TRAIN, TEST
