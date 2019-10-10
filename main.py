# libraries
from libraries import *

# local libraries
from parameters import *
from bound_maker import *
from Custom_layers import *
from functions import *
from model import *

# main body
# holla
model = east_net()

model = model_train(model, setweights=True, iteration=iteration)

result = predict(model)
data = result[0]
score = result[1]
results = result[2]

imgSave(mode='data', array=data)
imgSave(mode='score', array=score)
imgSave(mode='result', array=results)

arSave(mode='data', array=data)
arSave(mode='score', array=score)
arSave(mode='result', array=results)

visualise(data, score, results)
