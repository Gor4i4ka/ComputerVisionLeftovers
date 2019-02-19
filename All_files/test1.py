import numpy as np
from xgboost import XGBRegressor as XGB
from sklearn.metrics import accuracy_score as score
FRAME = 384
RGB = 3
SIZE = 250
PARTS = 2
CLASS = 2
PATH = 'D://IMGBUF1/'
DEPTH = 3
ESTIMATORS = 300
RATE = 0.05
#

SHAPE = (2, 2, 3)
SHAPE1 = (2,3)

ARR = np.zeros(SHAPE)
ARR1 = np.zeros(SHAPE1)
ARR2 = np.zeros(SHAPE1)

def proxy(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                arr[i][j][k] = i*arr.shape[1]*arr.shape[2] + j*arr.shape[2] + k

def proxy2d(arr, N):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
                arr[i][j] = i * arr.shape[1] + j + N

#proxy(ARR)
proxy2d(ARR1, 0)
proxy2d(ARR2, 6)

ARR[0] = ARR1
ARR[1] = ARR2

#print(ARR1)
#print(ARR2)

#print(ARR)
buf = ARR.flatten()
#print(buf.reshape((1,buf.shape[0])))
#print(ARR.reshape((2,2,3)))
A = np.arange(16)
A = A.reshape((4, 4))
B = np.arange(4) % 2
print(A)
print(B)
model = XGB(max_depth=DEPTH, n_estimators=ESTIMATORS, learning_rate=RATE, nthread=4)
model.fit(A, B)
Y_P = model.predict(A)
print(Y_P)
accuracy = score(B, Y_P.round())
print('TRAIN ACCURACY = ', accuracy*100, '%')

