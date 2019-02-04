from matplotlib import pyplot as py
from PIL import Image
import numpy as np
import glob
import os
import scipy.misc as rs

FRAME = 1024

folders = [
        '/home/gor4i4ka/imgDem/full',
        '/home/gor4i4ka/imgmem/full',
        '/home/gor4i4ka/imgScaJ',
        '/home/gor4i4ka/imgCleJ'
        ]

length = 0
for fold in folders:
    length += len(os.listdir(fold))

ARR = np.zeros((length, FRAME, FRAME, 3))
TAR = np.zeros((length))
ind = 0
FLD = 0

#DEM = 0; MEM = 1; SCA = 2; CLE = 3

for fold in folders:
    length = len(os.listdir(fold))
    for file in os.listdir(fold):
        try:
            img = Image.open(fold + '/' + file).convert('RGB')
        except:
            continue
        img1 = rs.imresize(img, (FRAME, FRAME))
        data = np.asarray(img1, dtype = 'int32')

        ARR[ind] = data
        TAR[ind] = FLD

        ind+=1
    FLD += 1
