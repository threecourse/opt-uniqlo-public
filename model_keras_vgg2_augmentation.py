import sys
sys.path.append(".")

import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
import h5py
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import util
from util_log import Logger
logger = Logger()
from model import ModelBase, ModelRunnerBase
from model_keras_util import Iterator
from model_keras_util_augmentation import ImageAugmentor
from util import Util

class ModelKerasVGG2Augmentation(ModelBase):

    def _get_model_structure(self):

        K.clear_session()

        # "structure"
        if self.prms["structure"] == "dense1":
            # basic
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = base_model.output
            x = Flatten()(x)
            x = Dense(1024, activation='relu')(x)
            inputs=base_model.input
            predictions = Dense(24, activation='softmax')(x)
        elif self.prms["structure"] == "dense_dropout":
            # with dropout
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

            x = base_model.output
            x = Flatten()(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='relu')(x)
            inputs = base_model.input
            predictions = Dense(24, activation='softmax')(x)

        elif self.prms["structure"] == "dense_freeze":
            # with freeze block1 and block2 layers
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = base_model.output
            x = Flatten()(x)
            x = Dense(1024, activation='relu')(x)
            inputs = base_model.input
            predictions = Dense(24, activation='softmax')(x)
            for i, layer in enumerate(base_model.layers):
                if i <= 6:
                    layer.trainable = False
        else:
            raise Exception

        model = Model(inputs=inputs, outputs=predictions)
        return model

    def train(self, prms, x_tr, y_tr, x_te, y_te):

        self.prms = prms
        # {"nb_epoch": 6, "batch_size":32, "lr":0.0001, "momentum":0.9, "patience":10, "decay":0.0, "nesterov":True}

        nb_epoch=self.prms["nb_epoch"]
        batch_size=self.prms["batch_size"]
        steps_per_epoch=self.prms["steps_per_epoch"]

        y_tr = np_utils.to_categorical(y_tr, num_classes=24)
        y_te = np_utils.to_categorical(y_te, num_classes=24)

        self.model = self._get_model_structure()
        self.model.compile(optimizer=SGD(lr=self.prms["lr"], momentum=self.prms["momentum"],
                           decay=self.prms["decay"], nesterov=self.prms["nesterov"]),
                      loss='categorical_crossentropy',
                      metrics=[metrics.categorical_crossentropy, metrics.categorical_accuracy])
        monitor = EarlyStopping(monitor="val_loss", patience=self.prms["patience"])
        # backup = ModelCheckpoint(filepath="../model/checkpoint/keras_{}.hd5".format(self.run_fold_name), save_best_only=True)

        logger.info("start keras training {}".format(self.run_fold_name))
        augmentor = ImageAugmentor(seed=self.prms["aug_seed"])
        g_tr = Iterator(x_tr, y_tr, batch_size=batch_size, shuffle=True, seed=71, batch_x_func=augmentor.transform_batch_train_image)
        self.hist = self.model.fit_generator(g_tr, steps_per_epoch=steps_per_epoch, epochs=nb_epoch,
                                             verbose=1, callbacks=[monitor], validation_data=(x_te, y_te))

        logger.info(self.hist.history)

    def train_without_validation(self, prms, x_tr, y_tr):
        raise NotImplementedError

    def save_model(self):
        Util.dumpc(self.prms, "../model/model/keras_{}_prms.pkl".format(self.run_fold_name))
        self.model.save("../model/model/keras_{}.hd5".format(self.run_fold_name), overwrite=True)

    def load_model(self):
        self.prms = Util.load("../model/model/keras_{}_prms.pkl".format(self.run_fold_name))
        self.model = keras.models.load_model("../model/model/keras_{}.hd5".format(self.run_fold_name))

    def predict(self, x_te):
        test_time_augmentation = self.prms["test_time_augmentation"]

        if test_time_augmentation >= 1:
            logger.info("test data augmentation prediction")
            augmentor = ImageAugmentor(seed=self.prms["aug_seed"])  # seed is initialized

            preds = []
            batch_size = 32
            for i_start in range(0, len(x_te), batch_size):
                i_end = min(i_start + batch_size, len(x_te))
                x_te_batch = x_te[i_start:i_end, :].copy()
                x_te_batch_augmented = augmentor.transform_batch_test_image(x_te_batch, n=test_time_augmentation)
                preds_batch = []
                for i in range(test_time_augmentation):
                    _x_te = x_te_batch_augmented[:, i, :]
                    preds_batch.append(self.model.predict_on_batch(_x_te))
                pred_batch = np.mean(np.array(preds_batch), axis=0)
                preds.append(pred_batch)
                if i_start % 320 == 0:
                    logger.info("predicted batch {}".format(i_start))
            pred = np.vstack(preds)
        else:
            pred = self.model.predict(x_te, batch_size=32)
        return pred
