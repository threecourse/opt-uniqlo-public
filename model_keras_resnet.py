import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
import h5py
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import util
from util_log import Logger
logger = Logger()
from model import ModelBase, ModelRunnerBase
from model_keras_util import Iterator

class ModelKerasResnet(ModelBase):

    def __init__(self, run_fold_name, num_classes=24):
        self.run_fold_name = run_fold_name
        self.num_classes = num_classes

    def _get_model_structure(self):

        K.clear_session()

        # "structure"
        if self.prms["structure"] == "dense_dropout":
            # with dropout
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = base_model.output
            x = Flatten()(x)
            x = Dropout(self.prms["dropout1"])(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(self.prms["dropout2"])(x)
            inputs = base_model.input
            predictions = Dense(self.num_classes, activation='softmax')(x)
        else:
            raise Exception

        model = Model(inputs=inputs, outputs=predictions)
        return model

    def train(self, prms, x_tr, y_tr, x_te, y_te, w_tr=None, w_te=None):

        self.prms = prms
        # {"nb_epoch": 6, "batch_size":32, "lr":0.0001, "momentum":0.9, "patience":10, "decay":0.0, "nesterov":True}

        nb_epoch=self.prms["nb_epoch"]
        batch_size=self.prms["batch_size"]
        steps_per_epoch=self.prms["steps_per_epoch"]

        y_tr = np_utils.to_categorical(y_tr, num_classes=self.num_classes)
        y_te = np_utils.to_categorical(y_te, num_classes=self.num_classes)

        self.model = self._get_model_structure()
        self.model.compile(optimizer=SGD(lr=self.prms["lr"], momentum=self.prms["momentum"],
                           decay=self.prms["decay"], nesterov=self.prms["nesterov"]),
                      loss='categorical_crossentropy',
                      metrics=[metrics.categorical_crossentropy, metrics.categorical_accuracy])
        monitor = EarlyStopping(monitor="val_loss", patience=self.prms["patience"])
        # backup = ModelCheckpoint(filepath="../model/checkpoint/keras_{}.hd5".format(self.run_fold_name), save_best_only=True)

        logger.info("start keras training {}".format(self.run_fold_name))
        g_tr = Iterator(x_tr, y_tr, batch_size=batch_size, shuffle=True, seed=71, w=w_tr)

        if w_te is None:
            validation_data = (x_te, y_te)
        else:
            print y_te[:5], w_te[:5]
            validation_data = (x_te, y_te, w_te)

        # TODO metrics is not weighted in keras monitoring
        self.hist = self.model.fit_generator(g_tr, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1, callbacks=[monitor], validation_data=validation_data)

        logger.info(self.hist.history)

    def train_without_validation(self, prms, x_tr, y_tr):
        raise NotImplementedError

    def save_model(self):
        self.model.save("../model/model/keras_{}.hd5".format(self.run_fold_name), overwrite=True)

    def load_model(self):
        self.model = keras.models.load_model("../model/model/keras_{}.hd5".format(self.run_fold_name))

    def predict(self, x_te):
        pred = self.model.predict(x_te, batch_size=32)
        return pred
