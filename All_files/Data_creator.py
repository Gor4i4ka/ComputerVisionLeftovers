from matplotlib import pyplot as py
from PIL import Image
import numpy as np
import glob
import os
import scipy.misc as rs
import pandas as pd

FRAME = 384
PART_NUM = 4
SIZE = 250
PART = 0

folders = [
    'D://IMGDEM',
    'D://IMGMEM',
    'D://IMGSCAN',
    'D://IMGCARD'
]


# DEM = 0; MEM = 1; SCA = 2; CLE = 3

# length = 0
# for fold in folders:
#    length += len(os.listdir(fold))

def data_create(F_PART, F_FOLDERS, F_SIZE):
    ARR = np.zeros((F_SIZE, FRAME, FRAME, 3))
    ind = 0
    FLD = 0

    for fold in F_FOLDERS:
        length = len(os.listdir(fold))
        PART_IND = 0
        for file in os.listdir(fold):
            if PART_IND < F_PART * F_SIZE:
                PART_IND += 1
                continue
            try:
                img = Image.open(fold + '/' + file).convert('RGB')
                #img.show()
            except:
                print(ind)
                continue

            img1 = rs.imresize(img, (FRAME, FRAME))

            if ind == F_SIZE:
                ind = 0
                #print('A')
                DST_D = 'D://IMGBUF1/data' + str(FLD) + 'part' + str(F_PART) + '.npy'
                print(DST_D)
                #print('B')
                np.save(DST_D, ARR)
                #print('C')
                break

            ARR[ind] = img1
            #print(ind)
            ind += 1

        FLD += 1
        print('CHECK', FLD)


for i in range(PART_NUM):
    data_create(i, folders, SIZE)
#data_create(3,folders,SIZE)