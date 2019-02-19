import Loader
import numpy as np
from xgboost import XGBRegressor as XGB
from sklearn.metrics import accuracy_score as score
from sklearn.linear_model import Ridge
#PARAMETERS
FRAME = 384
RGB = 3
SIZE = 250
PARTS = 2
CLASS = 4
PATH = 'D://IMGBUF1/'
ALPH = 1.0
RATE = 0.05
NUM = 64
BINS = np.zeros((NUM+1))
ITER = None
GRADE = 5
FL = True
for i in range(1, NUM+1):
    BINS[i] = NUM*i

#Loading data
X, Y = Loader.data_load(CLASS, PARTS, PATH)

#Using colour histograms if needed
X = Loader.histogram(X, BINS, NUM)
#Reshape and polynomize if needed
if FL:
    X = Loader.preproc(X, normalize=False, reshape= True)[0]
    print('DO ', X.shape)
    X = Loader.polynom(X, GRADE)
    print('POSLE', X.shape)
    X, TRAIN_IND, TEST_IND = Loader.preproc(X, reshape=False)
#preprocessing and data split if needed
if not FL:
    X, TRAIN_IND, TEST_IND = Loader.preproc(X)
print(X.shape)
print(X[TRAIN_IND].shape, X[TEST_IND].shape)
eval_set = [X[TEST_IND], Y[TEST_IND]]

#The Ridge
model = Ridge(alpha=ALPH, max_iter=ITER)
model.fit(X[TRAIN_IND], Y[TRAIN_IND])

#prediction
#X_TEST = X[TEST_IND]
Y_P = model.predict(X[TEST_IND])
accuracy = score(Y[TEST_IND], Y_P.round())
print('TEST ACCURACY = ', accuracy*100, '%')
