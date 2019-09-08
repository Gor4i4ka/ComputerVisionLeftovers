# libraries
from libraries import *

# local libraries
from parameters import *
from bound_maker import *
from Custom_layers import *
from functions import *

def east_net():

    #PVA NET

    main_input = Input(shape=(512, 512, 3), name='main_input')

    conv0 = Conv2D( filters=16,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    padding='same',
                    name='conv0')(main_input)
    conv0 = LeakyReLU(alpha=ALPHA_RE)(conv0)

    conv1 = Conv2D( filters=64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='same',
                    name='conv1')(conv0)
    conv1 = LeakyReLU(alpha=ALPHA_RE)(conv1)

    conv2 = Conv2D(filters=128,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   name='conv2')(conv1)
    conv2 = LeakyReLU(alpha=ALPHA_RE)(conv2)

    conv3 = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   name='conv3')(conv2)
    conv3 = LeakyReLU(alpha=ALPHA_RE)(conv3)

    conv4 = Conv2D(filters=384,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   name='conv4')(conv3)
    conv4 = LeakyReLU(alpha=ALPHA_RE)(conv4)
    # PVA_NET END
    # FEATURE MERGING BRANCH

    unpool1 = UpSampling2D(size=(2, 2))(conv4)

    concat1 = Concatenate(axis=-1)([unpool1, conv3])

    convM1_1 = Conv2D(filters=128,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      name='convM1_1')(concat1)
    convM1_1 = LeakyReLU(alpha=ALPHA_RE)(convM1_1)

    convM1_2 = Conv2D(filters=128,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      name='convM1_2')(convM1_1)
    convM1_2 = LeakyReLU(alpha=ALPHA_RE)(convM1_2)

    unpool2 = UpSampling2D(size=(2, 2)) (convM1_2)

    concat2 = Concatenate(axis=-1)([unpool2, conv2])

    convM2_1 = Conv2D(filters=64,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      name='convM2_1')(concat2)
    convM2_1 = LeakyReLU(alpha=ALPHA_RE)(convM2_1)

    convM2_2 = Conv2D(filters=64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      name='convM2_2')(convM2_1)
    convM2_2 = LeakyReLU(alpha=ALPHA_RE)(convM2_2)

    unpool3 = UpSampling2D(size=(2, 2))(convM2_2)

    concat3 = Concatenate(axis=-1)([unpool3, conv1])

    convM3_1 = Conv2D(filters=32,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      name='convM3_1')(concat3)
    convM3_1 = LeakyReLU(alpha=ALPHA_RE)(convM3_1)

    convM3_2 = Conv2D(filters=32,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      name='convM3_2')(convM3_1)
    convM3_2 = LeakyReLU(alpha=ALPHA_RE)(convM3_2)

    convM4 = Conv2D(filters=32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    name='convM4')(convM3_2)
    convM4 = LeakyReLU(alpha=ALPHA_RE)(convM4)

    score = Conv2D( filters=1,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='same',
                    name='conv_score'
                    ,activation=sigmoid
                    )(convM4)

    model = Model(inputs=main_input, outputs=score)

    model.summary()

    model.compile(loss=class_loss, optimizer=opt(learning_rate), metrics=[ms])

    return model
