# coding: utf-8
import os
import numpy as np
from keras import models
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, CuDNNLSTM, Embedding
from keras.utils import to_categorical
# from keras.preprocessing.sequence import pad_sequences
# from keras.optimizers import Adam, SGD


class BaseModel(object):
    CHECKPOINT_BASE_DIR = './model_checkpoints'

    def __init__(self, name, file_pattern='weights.{epoch:03d}.hdf5', input_size=1, output_size=3):
        self._name = name
        self._file_pattern = file_pattern
        self._model_dir = os.path.join(
            self.CHECKPOINT_BASE_DIR, name
        )
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        self._model = None
        self._input_size = input_size
        self._output_size = output_size

    def __unicode__(self):
        return self._name

    def checkpoint_path(self):
        return os.path.join(self._model_dir, self._file_pattern)

    def build_train_model(self, *args, **kwargs):
        raise NotImplementedError

    def build_predict_model(self, *args, **kwargs):
        raise NotImplementedError

    def load_model_weights(self, init_epoch=None):
        if init_epoch and self._model:
            weights_file = os.path.join(
                self._model_dir,
                self._file_pattern.format(epoch=init_epoch)
            )
            self._model.load_weights(weights_file)

    def data_generator(self, *args, **kwargs):
        raise NotImplementedError


class BenchMarkModel(BaseModel):
    def _build_model(self):
        model = models.Sequential()
        model.add(
            LSTM(
                32, return_sequences=True,
                input_shape=(None, 1),
                dropout=0.1, # recurrent_dropout=0.5,
                # activity_regularizer=keras.regularizers.l1(0.2)
            )
        )
        model.add(
            LSTM(
                32, return_sequences=True,
                dropout=0.1, # recurrent_dropout=0.5,
                # activity_regularizer=keras.regularizers.l1(0.2)
            )
        )
        model.add(
            LSTM(
                16, return_sequences=True,
                dropout=0.1, # recurrent_dropout=0.5,
                # activity_regularizer=keras.regularizers.l1(0.2)
            )
        )
        model.add(TimeDistributed(Dense(self._output_size, activation='softmax')))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        self._model = model
        return model

    def build_train_model(self):
        return self._build_model()

    def build_predict_model(self):
        return self._build_model()

    def data_generator(self, data_x, data_y):
        pairs = [
            (x, to_categorical(
                data_y[i], num_classes=self._output_size
            )) for i, x in enumerate(data_x)
        ]
    	while True:
    	    np.random.shuffle(pairs)
    	    for p in pairs:
    	        x_len = len(p[0])
    	        x = np.array(p[0]).reshape(1, x_len, self._input_size)
    	        y = np.array(p[1]).reshape(1, x_len, self._output_size)
                yield (x, y)


class BidirectionModel(BaseModel):
    def _build_model(self):
        model = models.Sequential()
        model.add(Embedding(24, 16))
        model.add(
            Bidirectional(CuDNNLSTM(
                16, return_sequences=True,
            ))
        )
        model.add(
            Bidirectional(CuDNNLSTM(
                16, return_sequences=True,
            ))
        )
        model.add(TimeDistributed(Dense(self._output_size, activation='softmax')))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        self._model = model
        return model

    def build_train_model(self):
        return self._build_model()

    def build_predict_model(self):
        return self._build_model()

    def data_generator(self, data_x, data_y, batch_size=1):
        pairs = [
            (x, to_categorical(
                data_y[i], num_classes=self._output_size
            )) for i, x in enumerate(data_x)
        ]
    	while True:
    	    np.random.shuffle(pairs)
    	    for p in pairs:
    	        x_len = len(p[0])
    	        x = np.array(p[0]).reshape(1, x_len)
    	        y = np.array(p[1]).reshape(1, x_len, self._output_size)
                yield (x, y)

