from model import ModelBase, ModelRunnerBase
import keras
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
from keras.optimizers import SGD, Adadelta, Adam
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, Activation
import pandas as pd
import numpy as np
from util import Util
from sklearn.preprocessing import StandardScaler
from keras.layers import merge, Conv2D, MaxPooling2D, AveragePooling2D, Input, Dense, Flatten, Activation, Reshape, Lambda
from keras.regularizers import l2 as regularizers_l2

class ModelKerasMLP(ModelBase):

    def _get_model_structure(self, input_dim, i_bag):

        seed = self.prms["seed"] if self.prms.has_key("seed") else 71
        np.random.seed(seed + i_bag)
        init = 'he_normal'
        activation = 'relu'
        num_classes = 24

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

        hidden_layers = self.prms["hidden_layers"]
        assert (hidden_layers in [0, 1, 2, 3])

        l2 = Util.get_item(self.prms, "weight_decay", None)
        if l2 is not None:
            kernel_regularizer = regularizers_l2(self.prms["weight_decay"])
        else:
            kernel_regularizer = None

        if hidden_layers == 0:
            # for testing
            inputs = Input(shape=(24,))
            predictions = Activation('softmax')(inputs)
            model = Model(inputs=inputs, outputs=predictions)
            print model.summary()
        else:
            model = Sequential()
            model.add(Dense(self.prms["h1"], input_dim=input_dim, init=init, kernel_regularizer=kernel_regularizer))
            model.add(Activation(activation))
            model.add(Dropout(self.prms["dropout1"]))
            if hidden_layers >= 2:
                model.add(Dense(self.prms["h2"], init=init, kernel_regularizer=kernel_regularizer))
                model.add(Activation(activation))
                model.add(Dropout(self.prms["dropout2"]))
            if hidden_layers >= 3:
                model.add(Dense(self.prms["h3"], init=init, kernel_regularizer=kernel_regularizer))
                model.add(Activation(activation))
                model.add(Dropout(self.prms["dropout3"]))
            model.add(Dense(num_classes, init=init))
            model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=[metrics.categorical_crossentropy, metrics.categorical_accuracy])

        return model

    def train(self, prms, x_tr, y_tr, x_te, y_te):
        self.prms = prms

        nb_epoch=self.prms["nb_epoch"]
        batch_size=self.prms["batch_size"]
        patience=self.prms["patience"]

        if self.prms["scaling"]:
            self.scaler = StandardScaler()
            self.scaler.fit(x_tr.astype(float))
            x_tr = self.scaler.transform(x_tr.astype(float))
            x_te = self.scaler.transform(x_te.astype(float))

        y_tr = np_utils.to_categorical(y_tr)
        y_te = np_utils.to_categorical(y_te)
        input_dim = x_tr.shape[1]

        self.models = []
        self.hists = []
        for i in range(self.prms["bags"]):
            model = self._get_model_structure(input_dim, i)
            monitor = EarlyStopping(monitor="val_loss", patience=patience)
            # backup = ModelCheckpoint(filepath="../model/checkpoint/keras_{}.hd5".format(self.run_fold_name), save_best_only=True)

            hist = model.fit(x_tr, y_tr,
                                       batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1,
                                       validation_data=(x_te, y_te),
                                       callbacks=[monitor])
            print hist.history
            self.models.append(model)
            self.hists.append(hist)

    def train_without_validation(self, prms, x_tr, y_tr):
        raise NotImplementedError

    def save_model(self):
        Util.dumpc(self.prms, "../model/model/keras_{}_prms.pkl".format(self.run_fold_name))
        if self.prms["scaling"]:
            Util.dumpc(self.scaler, "../model/model/keras_{}_scaler.pkl".format(self.run_fold_name))

        for i in range(self.prms["bags"]):
            self.models[i].save("../model/model/keras_{}_bag{}.hd5".format(self.run_fold_name, i), overwrite=True)

    def load_model(self):
        self.prms = Util.load("../model/model/keras_{}_prms.pkl".format(self.run_fold_name))
        if self.prms["scaling"]:
            self.scaler = Util.load("../model/model/keras_{}_scaler.pkl".format(self.run_fold_name))

        self.models = []
        for i in range(self.prms["bags"]):
            model = keras.models.load_model("../model/model/keras_{}_bag{}.hd5".format(self.run_fold_name, i))
            self.models.append(model)

    def predict(self, x_te):
        if self.prms["scaling"]:
            x_te = self.scaler.transform(x_te.astype(float))
        preds = []
        for i in range(self.prms["bags"]):
            pred = self.models[i].predict(x_te, batch_size=256)
            preds.append(pred)
        return np.array(preds).mean(axis=0)
