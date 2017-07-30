import sys
sys.path.append(".")

import h5py
import numpy as np
import pandas as pd
from util import Util
import re

from model import ModelBase, ModelRunnerBase

class ModelMix(ModelBase):

    def train(self, prms, x_tr, y_tr, x_te, y_te):
        pass

    def train_without_validation(self, prms, x_tr, y_tr):
        raise NotImplementedError

    def save_model(self):
        pass

    def load_model(self):
        pass

    def predict(self, x_te):
        return x_te

