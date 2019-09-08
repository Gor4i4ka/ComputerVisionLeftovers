# libraries
from libraries import *

# local libraries
from parameters import *
from Custom_layers import *

# random seed
np.random.seed(1000)


def prep(dt):
    dt = (dt - dt.mean(axis=0)) / dt.std(axis=0)
    dt = (dt - dt.min(axis=0)) / (dt.max(axis=0) - dt.min(axis=0))
    dt = np.nan_to_num(dt)
    return dt


def imgSave(mode, array):
    for i in range(array.shape[0]):
        if mode=='data':
            img = Image.fromarray(array[i].astype('uint8'), mode='RGB')
            img.save("D://IMGRESULTS/DATA/img" + str(i) + ".jpg")
        if mode=='score':
            array = array.reshape((array.shape[0], OUTPUT_SIZE, OUTPUT_SIZE))
            img = Image.fromarray(COEF * array[i].astype('uint8'), mode='L')
            img.save("D://IMGRESULTS/SCORE/img" + str(i) + ".jpg")
        if mode=='result':
            img = Image.fromarray(COEF * array[i].astype('uint8'), mode='L')
            img.save("D://IMGRESULTS/RESULTS/img" + str(i) + ".jpg")
        if mode=='bound':
            print(i)
            img = Image.fromarray(array[i].astype('uint8'), mode='L')
            img.save("D://IMGRESULTS/BOUND/img" + str(i) + ".jpg")


def opt(lr):
    return keras.optimizers.adam(lr=lr)


def class_loss(y_true, y_pred):
    y_tr = tf.reshape(y_true, ((BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 1)))
    y_pr = tf.reshape(y_pred, ((BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 1)))

    B = 1 - K.mean(y_tr)
    eps = 1e-4
    cl_loss = (-1) * B * y_tr * K.log(y_pr + eps)
    cl_loss1 = (1 - B) * (1 - y_tr) * K.log(1 - y_pr + eps)
    cl_final = K.sum(cl_loss - cl_loss1)
    return cl_final


def model_train(model, setweights=False, path_weights=weigh_path, iteration=0):
    if setweights:
        print("WEIGHTS LOADED FROM: ", path_weights + str(iteration - 1))
        model.load_weights(path_weights + str(iteration - 1))
    record_ar = np.zeros((total_batches))
    min_ar = np.zeros((total_batches))
    val_score = 0
    val_visit_flag = False
    counter = 0
    cur_batch_pack = 0
    cur_batch = 0
    for i in range(EP * total_batches):
        if i == 0:
            dt = np.load(
                "D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(1) + ".npy")
            #print(
             #   "D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(1 + batch_am * cur_batch_pack) + ".npy")
            sc = np.load(
                "D://DATACOCOTEST" + str(BATCH_SIZE) + "/SCORE/score_part" + str(1) + ".npy")
            sc = sc.reshape((sc.shape[0], OUTPUT_SIZE, OUTPUT_SIZE, 1))

            for ar in range(2, batch_am + 1):
                #print("D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(ar) + ".npy")
                bufd = np.load(
                    "D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(ar) + ".npy")
                bufs = np.load(
                    "D://DATACOCOTEST" + str(BATCH_SIZE) + "/SCORE/score_part" + str(ar) + ".npy")
                bufs = bufs.reshape((bufs.shape[0], OUTPUT_SIZE, OUTPUT_SIZE, 1))
                dt = np.concatenate((dt, bufd))
                sc = np.concatenate((sc, bufs))
                del(bufd)
                del(bufs)
                gc.collect()

            dt = prep(dt)

        if i % batch_am == 0 and i > 0:
            cur_batch_pack += 1

            if cur_batch_pack * batch_am >= total_batches:
                cur_batch_pack = 0

            dt = np.load(
                "D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(1 + batch_am * cur_batch_pack) + ".npy")
            #print("D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(1 + batch_am * cur_batch_pack) + ".npy")
            sc = np.load(
                "D://DATACOCOTEST" + str(BATCH_SIZE) + "/SCORE/score_part" + str(1 + batch_am * cur_batch_pack) + ".npy")
            sc = sc.reshape((sc.shape[0], OUTPUT_SIZE, OUTPUT_SIZE, 1))

            for ar in range(2 + batch_am * cur_batch_pack, batch_am * (cur_batch_pack + 1) + 1):
                #print("D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(ar) + ".npy")
                bufd = np.load(
                    "D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(ar) + ".npy")
                bufs = np.load(
                    "D://DATACOCOTEST" + str(BATCH_SIZE) + "/SCORE/score_part" + str(ar) + ".npy")
                bufs = bufs.reshape((bufs.shape[0], OUTPUT_SIZE, OUTPUT_SIZE, 1))
                dt = np.concatenate((dt, bufd))
                sc = np.concatenate((sc, bufs))
                del(bufd)
                del(bufs)
            dt = prep(dt)
        score_feed = np.copy(sc[(cur_batch) * BATCH_SIZE: (cur_batch + 1) * BATCH_SIZE])
        for time in range(times_batch):
            loss = model.train_on_batch(x=dt[(cur_batch) * BATCH_SIZE: (cur_batch + 1) * BATCH_SIZE], y=score_feed)

        if i > 0 and i % val_per == 0:
            if val_visit_flag == False:
                val_visit_flag = True
                for it in range(val_batch):
                    dt_val = np.load("D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(1 + total_batches + it) + ".npy")
                    dt_val = prep(dt_val)
                    sc_val = np.load("D://DATACOCOTEST" + str(BATCH_SIZE) + "/SCORE/score_part" + str(1 + total_batches + it) + ".npy")
                    sc_val = sc_val.reshape((sc_val.shape[0], OUTPUT_SIZE, OUTPUT_SIZE, 1))
                    loss_val = model.test_on_batch(x=dt_val, y=sc_val)
                    val_score += loss_val[0]
                    val_min_weights = model.get_weights()
                print("FIRST VAL_SCORE ", val_score)
            else:
                loss_sum = 0
                for it in range(val_batch):
                    dt_val = np.load("D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(1 + total_batches + it) + ".npy")
                    dt_val = prep(dt_val)
                    sc_val = np.load("D://DATACOCOTEST" + str(BATCH_SIZE) + "/SCORE/score_part" + str(1 + total_batches + it) + ".npy")
                    sc_val = sc_val.reshape((sc_val.shape[0], OUTPUT_SIZE, OUTPUT_SIZE, 1))
                    loss_val = model.test_on_batch(x=dt_val, y=sc_val)
                    loss_sum += loss_val[0]
                if val_score > loss_sum:
                    print("NEW VALIDATION MINIMUM DETECTED")
                    val_score = loss_sum
                    val_min_weights = model.get_weights()
                print("VAL_MIN_LOSS ", val_score, " CUR_VAL_LOSS ", loss_sum)
        if i < total_batches:
            if i == 0 or min_loss[0] > loss[0]:
                print("NEW MINIMUM. LESS THAN TOTAL BATCHES.")
                equality_counter = 0
                min_loss = loss
                min_weights = model.get_weights()

            min_ar[i % total_batches] = loss[0]
            record_ar[i % total_batches] = loss[0]

        if i >= total_batches:
            if record_ar[i % total_batches] == loss[0]:
                equality_counter += 1
            else:
                equality_counter = 0
            record_ar[i % total_batches] = loss[0]
            if min_ar.sum() > record_ar.sum():
                print("NEW MINIMUM. MORE THAN TOTAL BATCHES.")
                min_weights = model.get_weights()
                min_ar = np.copy(record_ar)
        if equality_counter == max_equal:
            print("STAGNATION DETECTED. NOISE ADDED")
            min_wei_clone = min_weights
            for el in range(len(min_wei_clone)):
                ar = min_wei_clone[el]
                shape = ar.shape
                ar = ar.reshape(-1)
                for slot in range(ar.shape[0]):
                    ar[slot] += random.random() * ar[slot] / share
                min_wei_clone[el] = ar.reshape(shape)
            model.set_weights(min_wei_clone)
        print("CUREBENCH = ", cur_batch, " CURE_PACK = ", cur_batch_pack, " TOTAL_BATCH ", i % total_batches,
              " COUNTER ", counter,  " MIN ", min_ar.sum(), " REC ", record_ar.sum(), " LOSS&MET ", loss)
        cur_batch += 1
        counter += 1
        if cur_batch > batch_am - 1:
            cur_batch = 0

    PATH = 'D://IMGRESULTS/ARRAYS/'
    model.set_weights(val_min_weights)
    model.save_weights(filepath=PATH + "val_weights" + str(iteration))
    model.set_weights(min_weights)
    model.save_weights(filepath=PATH + "train_weights" + str(iteration))
    return model


def predict(model):
    # SAVING DATA AND SCORE
    dt = np.load("D://DATACOCOTEST" + str(BATCH_SIZE) + "/DATA/data_part" + str(test_batch) + ".npy")
    sc = np.load("D://DATACOCOTEST" + str(BATCH_SIZE) + "/SCORE/score_part" + str(test_batch) + ".npy")
    dt1 = np.copy(dt)

    # PREDICTING
    dt = prep(dt)
    res = (model.predict(dt))

    # GETTING THE RESULT, SAVING IT
    ressc = res[:, :, :, 0]
    ressc = np.where(abs(ressc - 1) < binarization_thr, 1, 0)
    ressc = ressc.reshape((BATCH_SIZE, 128, 128))

    return dt1, sc, ressc


def visualise (data, score, results):
    # VISAULISATION FOR COMPARISON

    # data
    DT = data[img]
    DT = Image.fromarray(DT.astype('uint8'), 'RGB')
    DT.show()

    # score
    SC = COEF * score[img].astype('uint8')
    SC = SC.reshape((128, 128))
    SC = Image.fromarray(SC, 'L')
    SC.show()

    # result
    RES = COEF * results[img].astype('uint8')
    RES = Image.fromarray(RES, 'L')
    RES.show()

    return


def arSave(mode, array):
    PATH = 'D://IMGRESULTS/ARRAYS/'
    if mode == 'data':
        PATH = PATH + 'data.npy'
    if mode == 'score':
        PATH = PATH + 'score.npy'
    if mode == 'result':
        PATH = PATH + 'result.npy'
    np.save(PATH, array)
    return