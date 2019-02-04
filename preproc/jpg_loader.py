from matplotlib import pyplot as py
from PIL import Image
import numpy as np
import glob
import os
import scipy.misc as rs

img = Image.open("/home/gor4i4ka/imgDem/full/0700.jpg").convert('RGB')
data = np.asarray(img, dtype='int32')
img1 = rs.imresize(img, (1024, 1024))
data1 = np.asarray(img1, dtype = 'int32')

print(data.shape, '  ', data1.shape)

#print(data)
#print(type(data))
#print(data.shape)
#for i in range(data.shape[0]):
#    for j in range(data.shape[1]):
#        if data[i][j][0] < 200 or data[i][j][1] < 200 or data[i][j][2] < 200:
#            print('row: ',i, ' pil ',j, ' ', data[i][j])


fig = py.figure()
ax = fig.add_subplot(111)
ax.imshow(img)
py.show()

fig = py.figure()
ax = fig.add_subplot(111)
ax.imshow(img1)
py.show()

folders = [
        #'/home/gor4i4ka/imgDem/full',
        #'/home/gor4i4ka/imgmem/full',
        #'/home/gor4i4ka/imgScaJ',
        #'/home/gor4i4ka/imgCleJ'
]
#fold = '/home/gor4i4ka/imgmem/full'
c = 0
hgt = 0
wdt = 0
for fold in folders:
    for file in os.listdir(fold):
        #filename = os.fsdecode(file)
        #print(file)
        try:
            img = Image.open(fold + '/' + file).convert('RGB')
        except:
            continue
        data = np.asarray(img, dtype = 'int32')
        hgt += data.shape[0]
        wdt += data.shape[1]
        c += 1
        #print(data.shape, ' ', file)
    hgt = hgt/c
    wdt = wdt/c
    print(hgt, ' ', wdt)
    c =0
    hgt = 0
    wdt = 0