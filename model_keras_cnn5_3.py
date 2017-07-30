import sys
sys.path.append(".")

import keras
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import util
from util_log import Logger
logger = Logger()
from model import ModelBase, ModelRunnerBase
from model_keras_util import Iterator
from keras.layers import merge, Conv2D, MaxPooling2D, AveragePooling2D, Input, Dense, Flatten, Activation, Reshape, Lambda, BatchNormalization, Concatenate
from keras.models import Sequential
from keras.layers.merge import Multiply
from util import Util
import h5py
from model_keras_util_augmentation import ImageAugmentor
from keras.applications.resnet50 import identity_block, conv_block

class ModelKerasCNN5_3(ModelBase):

    def _get_model_structure(self, i_bag):

        inputs = Input(shape=(64, 64, 3))
        # inputs2 = Input(shape=(64, 64))

        # parameters
        ch1 = self.prms["ch1"]
        k1 = self.prms["k1"]
        # k1_2 = self.prms["k1_2"]
        strides1 = self.prms["strides1"]
        bn1 = self.prms["bn1"]
        dropout1 = self.prms["dropout1"]
        ch2 = self.prms["ch2"]

        x = Conv2D(ch1, kernel_size=(k1, k1))(inputs)
        x = Activation('relu')(x)
        x = Conv2D(ch1, kernel_size=(k1, k1), strides=(strides1, strides1))(x)
        if bn1 > 0:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout1)(x)

        # Resblock
        if self.prms["structure"] == 1:
            x = conv_block(x, 2, [128, 128, ch2], stage=2, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2], stage=2, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2], stage=2, block='c')

            x = conv_block(x, 2, [128, 128, ch2 * 2], stage=3, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2], stage=3, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2], stage=3, block='c')
        if self.prms["structure"] == 0:
            x = conv_block(x, 2, [128, 128, ch2], stage=2, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2], stage=2, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2], stage=2, block='c')

        elif self.prms["structure"] == 2:
            x = conv_block(x, 2, [128, 128, ch2], stage=2, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2], stage=2, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2], stage=2, block='c')

            x = conv_block(x, 2, [128, 128, ch2 * 2], stage=3, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2], stage=3, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2], stage=3, block='c')

            x = conv_block(x, 2, [128, 128, ch2 * 2 * 2], stage=4, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2 * 2], stage=4, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2 * 2] , stage=4, block='c')

        elif self.prms["structure"] == 3:
            x = conv_block(x, 2, [128, 128, ch2], stage=2, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2], stage=2, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2], stage=2, block='c')

            x = conv_block(x, 2, [128, 128, ch2 * 2], stage=3, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2], stage=3, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2], stage=3, block='c')

            x = conv_block(x, 2, [128, 128, ch2 * 2 * 2], stage=4, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2 * 2], stage=4, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2 * 2] , stage=4, block='c')

            x = conv_block(x, 2, [128, 128, ch2 * 2 * 2 * 2], stage=5, block='a')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2 * 2 * 2], stage=5, block='b')
            x = identity_block(x, kernel_size=2, filters=[64, 64, ch2 * 2 * 2 * 2] , stage=5, block='c')

        elif self.prms["structure"] == 4:
            x = conv_block(x, 3, [128, 128, ch2], stage=2, block='a')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2], stage=2, block='b')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2], stage=2, block='c')

            x = conv_block(x, 3, [128, 128, ch2 * 2], stage=3, block='a')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2 * 2], stage=3, block='b')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2 * 2], stage=3, block='c')

        elif self.prms["structure"] == 5:
            x = conv_block(x, 3, [128, 128, ch2], stage=2, block='a')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2], stage=2, block='b')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2], stage=2, block='c')

            x = conv_block(x, 3, [128, 128, ch2 * 2], stage=3, block='a')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2 * 2], stage=3, block='b')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2 * 2], stage=3, block='c')

            x = conv_block(x, 3, [128, 128, ch2 * 2 * 2], stage=4, block='a')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2 * 2 * 2], stage=4, block='b')
            x = identity_block(x, kernel_size=3, filters=[64, 64, ch2 * 2 * 2] , stage=4, block='c')

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        predictions = Dense(24, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)
        print model.summary()

        return model

    def train(self, prms, x_tr, y_tr, x_te, y_te):

        x_tr = x_tr[0]
        x_te = x_te[0]

        self.prms = prms

        nb_epoch=self.prms["nb_epoch"]
        batch_size=self.prms["batch_size"]
        steps_per_epoch=self.prms["steps_per_epoch"]

        y_tr = np_utils.to_categorical(y_tr, num_classes=24)
        y_te = np_utils.to_categorical(y_te, num_classes=24)

        self.models = []
        self.hists = []

        K.clear_session() # to save model properly

        # optimizer
        optimizer_type = Util.get_item(self.prms, "optimizer", "sgd")
        if optimizer_type == "sgd":
            optimizer = SGD(lr=self.prms["lr"], momentum=self.prms["momentum"], nesterov=True)
        elif optimizer_type == "adadelta":
            optimizer = Adadelta(lr=self.prms["lr"], rho=self.prms["rho"])
        elif optimizer_type == "adam":
            optimizer = Adam(lr=self.prms["lr"], beta_1=self.prms["beta_1"])
        else:
            raise Exception

        augmentor = ImageAugmentor(seed=self.prms["aug_seed"])
        for i in range(self.prms["bags"]):
            model = self._get_model_structure(i)
            model.compile(optimizer=optimizer,
                               loss='categorical_crossentropy',
                               metrics=[metrics.categorical_crossentropy, metrics.categorical_accuracy])
            monitor = EarlyStopping(monitor="val_loss", patience=self.prms["patience"])

            logger.info("start keras training {} {} ".format(self.run_fold_name, i))

            g_tr = Iterator(x_tr, y_tr, batch_size=batch_size, shuffle=True, seed=71,
                            batch_x_func=augmentor.transform_batch_train_image)
            hist = model.fit_generator(g_tr, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1,
                                       callbacks=[monitor], validation_data=(x_te, y_te))
            self.models.append(model)
            self.hists.append(hist)
            print hist.history

    def train_without_validation(self, prms, x_tr, y_tr):
        raise NotImplementedError

    def save_model(self):
        Util.dumpc(self.prms, "../model/model/keras_{}_prms.pkl".format(self.run_fold_name))
        for i in range(self.prms["bags"]):
            self.models[i].save("../model/model/keras_{}_bag{}.hd5".format(self.run_fold_name, i), overwrite=True)

    def load_model(self):
        self.prms = Util.load("../model/model/keras_{}_prms.pkl".format(self.run_fold_name))

        self.models = []
        for i in range(self.prms["bags"]):
            # work around bug ????
            # https://github.com/fchollet/keras/issues/4044
            model_path = "../model/model/keras_{}_bag{}.hd5".format(self.run_fold_name, i)
            with h5py.File(model_path, 'a') as f:
                if 'optimizer_weights' in f.keys():
                    del f['optimizer_weights']
            model = keras.models.load_model(model_path)
            self.models.append(model)

    def predict(self, x_te):
        x_te = x_te[0]

        test_time_augmentation = self.prms["test_time_augmentation"]

        if test_time_augmentation >= 1:
            logger.info("test data augmentation prediction")
            augmentor = ImageAugmentor(seed=self.prms["aug_seed"])  # seed is initialized

            preds = []
            batch_size = 256
            for i_start in range(0, len(x_te), batch_size):
                i_end = min(i_start + batch_size, len(x_te))
                x_te_batch = x_te[i_start:i_end, :].copy()
                x_te_batch_augmented = augmentor.transform_batch_test_image(x_te_batch, n=test_time_augmentation)
                preds_batch = []
                for i_bag in range(self.prms["bags"]):
                    for i in range(test_time_augmentation):
                        _x_te = x_te_batch_augmented[:, i, :]
                        preds_batch.append(self.models[i_bag].predict_on_batch(_x_te))
                pred_batch = np.mean(np.array(preds_batch), axis=0)
                preds.append(pred_batch)
                if i_start % 320 == 0:
                    logger.info("predicted batch {}".format(i_start))
            pred = np.vstack(preds)
            return pred
        else:
            preds = []
            for i in range(self.prms["bags"]):
                pred = self.models[i].predict(x_te, batch_size=256)
                preds.append(pred)
            return np.array(preds).mean(axis=0)


