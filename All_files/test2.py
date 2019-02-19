from matplotlib import pyplot as py
from PIL import Image
import numpy as np
import glob
import os
import scipy.misc as rs
import pandas as pd

FRAME = 1024

#IMG = Image.open('/home/gor4i4ka/exp/0001.jpg').convert('RGB')
#img1 = rs.imresize(IMG, (FRAME, FRAME))
#IMG1 = Image.fromarray(img1.astype('uint8'), 'RGB')
#IMG1.show()
#IMG1.save('/home/gor4i4ka/exp/LOLKA.PNG', 'PNG')

FRAME = 1024
SIZE = 500
PART = 0

folders = [
    'D://IMGDEM',
    #'/home/gor4i4ka/imgScaJ',
    #'/home/gor4i4ka/imgCleJ'
]


# DEM = 0; MEM = 1; SCA = 2; CLE = 3

# length = 0
# for fold in folders:
#    length += len(os.listdir(fold))

def data_create(F_FOLDERS):
    ind = 0

    for fold in F_FOLDERS:
        length = len(os.listdir(fold))
        #print('B')
        for file in os.listdir(fold):
            try:
                #print('C')
                img = Image.open(fold + '/' + file).convert('RGB')
                img1 = rs.imresize(img, (FRAME, FRAME))
                IMG1 = Image.fromarray(img1.astype('uint8'), 'RGB')
                #print('D')
                IMG1.save('/home/gor4i4ka/imgDemCon/' + str(ind) + '.PNG', 'PNG')
                #IMG1.save('/home/gor4i4ka/imgCleCON/' + ind, 'PNG')
                #IMG1.save('/home/gor4i4ka/imgScaCON/' + ind, 'PNG')
                ind += 1
                print(ind)
            except:
                continue
print('A')
data_create(folders)