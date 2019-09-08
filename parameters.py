# Constants
###########
# The batch size for training and label creating
BATCH_SIZE = 24

# learning rate of gradient descent, initial = 0.001
learning_rate = 0.0005

# Amount of epochs to train
EP = 10

# How many times model trains on a batch during an epoch
times_batch = 3
# How many batches to simultaneously load into RAM
batch_am = 2

# How many batches used for training
total_batches = batch_am * 250

# how many times you already subsequently trained the model
iteration = 8

# The parameter for how light you want to see the score_map results. range:(1, 255)
COEF = 255

# RGB. 3 colours.
RGB = 3

# The path to the place you store all COCO-text related data including created numpy arrays
PATH_TEST = 'D://'

# Leaky ReLU slide coefficient
ALPHA_RE = 0.3

# Noise cube parameter. The higher the share the smaller the noise.
share = 1000

# how many epochs loss can stagnate
max_equal = 1

# validation set check period
val_per = 50

# validation set size in batches
val_batch = 5

#COCO_TEXT_LEN
coco_len = 14323

# Input frame for the model
INPUT_SIZE = 512

# Output frame for the model
OUTPUT_SIZE = 128

# Input / output frame sizes
FACTOR = int(INPUT_SIZE / OUTPUT_SIZE)

# current image from the predicted batch for both main:visualise part and bound_maker to show
img = 2

weigh_path = 'D://IMGRESULTS/ARRAYS/train_weights'

# The binarization threshold applied to predicted results
binarization_thr = 0.2

# Amount of images fo COCO_LABEL_CREATOR to create for training in the form of numpy arrays.
IMG_AM = coco_len

# the batch to test model prediction
test_batch = 20

# The line_thickness of bounding boxes made by bound_maker
line_thickness = 1

# COCO-text annotations path
PATH_COCO = PATH_TEST + 'DATASETS/COCO_Text.json'

# COCO-text training part data path
PATH_DATA = PATH_TEST + 'DATASETS/TRAIN/train2014'

# The paths according to which numpy arrays will be saved
# requires:
# existance of THE directory PATH_TEST + 'DATACOCOTEST' + str(BATCH_SIZE) + '/'
# subdirectories SCORE, BOUND, DATA
PATH_SAVE = PATH_TEST + 'DATACOCOTEST' + str(BATCH_SIZE) + '/'