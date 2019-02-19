from matplotlib import pyplot as py
from PIL import Image
import numpy as np
import glob
import os
import scipy.misc as rs
import pandas as pd

PATH = 'D://IMGBUF1/data3part3.npy'
FRAME = 1024
FRAME1 = 384
F_SIZE = 2
RGB = 3

#img = Image.open("D://IMGDEM/0002.jpg").convert('RGB')
#img2 = Image.open("/home/gor4i4ka/imgmem/full/0010.jpg").convert('RGB')

#img1 = rs.imresize(img, (FRAME1, FRAME1))
#img2 = rs.imresize(img2, (FRAME, FRAME))
#IMG = Image.fromarray(img1.astype('uint8'), 'RGB')
#IMG.show()

#np.save(PATH, img1)
#print(type(img1), img1.shape)
#ARR = np.zeros((F_SIZE, FRAME, FRAME, RGB))
#print(type(ARR), ARR.shape)
#ARR[0] = img1
#ARR[1] = img2
#print(type(ARR[0]), ARR[0].shape)
#np.save(PATH, ARR)

BUF = np.load(PATH)
print(type(BUF), BUF.shape)
KEK = BUF[249]
#for v,i,j,k in np.ndindex(ARR.shape):
 #   if ARR[v][i][j][k] - BUF[v][i][j][k] != 0:
  #      print('XUI')
#print('AUE', img1.dtype, ARR[0].dtype)
IMG1 = Image.fromarray(KEK.astype('uint8'), 'RGB')
IMG1.show()


#A = np.load(PATH)[0]
#IMG = Image.fromarray(A, 'RGB')

#IMG.show()


