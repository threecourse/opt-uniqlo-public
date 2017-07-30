import numpy as np
import threading
from keras import backend as K

class Iterator(object):

    def __init__(self, x, y, batch_size, shuffle, seed, w=None, batch_x_func=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()

        if isinstance(x, list):
            self.x = [np.asarray(xx, dtype=K.floatx()) for xx in x]
            self.n = x[0].shape[0]
        else:
            self.x = np.asarray(x, dtype=K.floatx())
            self.n = x.shape[0]

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None

        if w is not None:
            self.w = np.asarray(w)
        else:
            self.w = None

        self.index_generator = self._flow_index(self.n, batch_size, shuffle, seed)
        self.batch_x_func = batch_x_func

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                               current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def get_batch_x(self, x, index_array, current_batch_size):
        batch_x = np.zeros(tuple([current_batch_size] + list(x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            _x = x[j]
            batch_x[i] = _x
        return batch_x

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        if isinstance(self.x, list):
            batch_x = [self.get_batch_x(x, index_array, current_batch_size) for x in self.x]
        else:
            batch_x = self.get_batch_x(self.x, index_array, current_batch_size)

        if self.batch_x_func is not None:
            batch_x = self.batch_x_func(batch_x)

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]

        if self.w is None:
            return batch_x, batch_y

        batch_w = self.w[index_array]
        return batch_x, batch_y, batch_w

