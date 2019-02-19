from matplotlib import pyplot as py
from PIL import Image
import numpy as np
import glob
import os
import scipy.misc as rs
import pandas as pd
from sklearn.model_selection import train_test_split as sk1
from sklearn.metrics import accuracy_score as sk2
from sklearn import preprocessing
from random import randint
K = np.arange(16) + 1
A = np.arange(16)
B = A.reshape(2, 2, 2, 2)
#print('B')
#print(B)
#print(preprocessing.normalize(B, norm='max'))
C = B.reshape(2, 8)
D = C.T
#print('C')
#print(D)
#print((preprocessing.normalize(D, norm='max')).T)
#print(np.prod(K))
M1, M2 = sk1(A, test_size=0.25, random_state=7)
print(M1,M2)