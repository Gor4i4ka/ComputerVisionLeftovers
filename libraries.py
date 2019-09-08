# LIBRARIES
import gc
import keras
import random
from keras.layers import Conv2D, LeakyReLU, Input, Concatenate, UpSampling2D
from keras.models import Model
from keras.metrics import mean_squared_error as ms
import numpy as np
import tensorflow as tf
from keras.activations import sigmoid
from keras.layers import Layer
from PIL import Image
from tensorflow.python.keras import backend as K
import COCO_TEXT_API
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import json
from PIL import Image
import scipy.misc as rs
import copy
import math