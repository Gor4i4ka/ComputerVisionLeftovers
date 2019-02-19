import Loader
import numpy as np
from xgboost import XGBRegressor as XGB
from sklearn.metrics import accuracy_score as score
#PARAMETERS
FRAME = 384
RGB = 3
SIZE = 250
PARTS = 2
CLASS = 4
PATH = 'D://IMGBUF1/'
DEPTH = 3
ESTIMATORS = 300
RATE = 0.05
NUM = 32
BINS = np.zeros((NUM+1))
for i in range(1, NUM+1):
    BINS[i] = NUM*i


#Loading data
X, Y = Loader.data_load(CLASS, PARTS, PATH)

#Using colour histograms if needed
X = Loader.histogram(X, BINS, NUM)

#preprocessing and data split if needed
X, TRAIN_IND, TEST_IND = Loader.preproc(X)
print(X.shape)
print(X[TRAIN_IND].shape, X[TEST_IND].shape)
eval_set = [X[TEST_IND], Y[TEST_IND]]
#Creating and fitting the model
model = XGB(max_depth=DEPTH, n_estimators=ESTIMATORS, learning_rate=RATE, nthread=4)
model.fit(X[TRAIN_IND], Y[TRAIN_IND])
NANI = np.copy(X[TEST_IND])
#Predictions for train data
#Y_P = model.predict(X[TRAIN_IND])
#accuracy = score(Y[TRAIN_IND], Y_P.round())
#print('TRAIN ACCURACY = ', accuracy*100, '%')
#Predictions for test data
Y_P = model.predict(NANI)
accuracy = score(Y[TEST_IND], Y_P.round())
print('TEST ACCURACY = ', accuracy*100, '%')


