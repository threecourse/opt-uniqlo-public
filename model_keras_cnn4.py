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
from keras.layers import merge, Conv2D, MaxPooling2D, AveragePooling2D, Input, Dense, Flatten, Activation, Reshape, Lambda, BatchNormalization
from keras.models import Sequential
from keras.layers.merge import Multiply
from util import Util
import h5py

class ModelKerasCNN4(ModelBase):

    def _get_model_structure(self, i_bag):

        inputs = Input(shape=(64, 64, 3))
        x = Conv2D(self.prms["ch1"], kernel_size=(1, 1), strides=(1, 1))(inputs)
        x = Activation('relu')(x)
        x = Conv2D(self.prms["ch2"], kernel_size=(1, 1), strides=(1, 1))(x)
        x = Activation('relu')(x)

        assert(self.prms["filters"] in [2, 3])
        if self.prms["filters"] == 2:
            x = Reshape((-1, self.prms["ch2"]))(x)

        if self.prms["filters"] == 3:
            x = Conv2D(self.prms["ch3"], kernel_size=(1, 1), strides=(1, 1))(x)
            x = Activation('relu')(x)
            x = Reshape((-1, self.prms["ch3"]))(x)

        inputs2 = Input(shape=(64, 64))
        x2 = Reshape((4096, 1))(inputs2)
        x = Multiply()([x, x2])

        def mean(x):
            x = K.mean(x, axis=1, keepdims=True)
            return x

        def mean_output_shape(input_shape):
            return tuple((input_shape[0], input_shape[2]))

        x = Lambda(mean, output_shape=mean_output_shape)(x)
        x = Dropout(self.prms["dropout1"])(x)
        x = Dense(self.prms["d1"])(x)
        if self.prms["bn"] >= 1:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.prms["dropout2"])(x)
        x = Dense(self.prms["d2"])(x)
        if self.prms["bn"] >= 2:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.prms["dropout3"])(x)
        predictions = Dense(24, activation='softmax')(x)

        model = Model(inputs=[inputs, inputs2], outputs=predictions)
        # print model.summary()

        return model

    def train(self, prms, x_tr, y_tr, x_te, y_te):

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

        for i in range(self.prms["bags"]):
            model = self._get_model_structure(i)
            model.compile(optimizer=optimizer,
                               loss='categorical_crossentropy',
                               metrics=[metrics.categorical_crossentropy, metrics.categorical_accuracy])
            monitor = EarlyStopping(monitor="val_loss", patience=self.prms["patience"])

            logger.info("start keras training {} {} ".format(self.run_fold_name, i))

            g_tr = Iterator(x_tr, y_tr, batch_size=batch_size, shuffle=True, seed=71)
            hist = model.fit_generator(g_tr, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1,
                                       callbacks=[monitor], validation_data=(list(x_te), y_te))
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
        preds = []
        for i in range(self.prms["bags"]):
            pred = self.models[i].predict(x_te, batch_size=256)
            preds.append(pred)
        return np.array(preds).mean(axis=0)
