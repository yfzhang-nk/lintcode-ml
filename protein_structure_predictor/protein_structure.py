# coding: utf-8
import numpy as np
from keras.callbacks import ModelCheckpoint, Callback

from nn_models import (
    BenchMarkModel, BidirectionModel, EncoderDecoderModel,
    LatestEncoderDecoderModel, AttentionEncoderDecoderModel,
)

AMIDOGEN = [
    'A',
    'R',
    'N',
    'D',
    'C',
    'Q',
    'E',
    'G',
    'H',
    'I',
    'L',
    'K',
    'M',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'U',
    'V',
    'X',
    'Y',
    'Z',
]
SECSTR = ['H', 'E', 'C']

INPUT_SIZE = 1
OUTPUT_SIZE = 3

def read_data(filename):
    features, labels = [], []
    with open(filename, "r") as f:
        lines = f.readlines()
        for idx, val in enumerate(lines):
            val = val.strip()
            if idx % 4 == 1:
                features.append(val)
            elif idx % 4 == 3:
                labels.append(val)
    return features, labels


def read_test_data(filename):
    features = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for idx, val in enumerate(lines):
            val = val.strip()
            if idx % 2 == 1:
                features.append(val)
    return features


def preprocess_data(features, labels=[]):
    x = []
    y = []
    for f in features:
        x.append([AMIDOGEN.index(s) for s in f])
    for l in labels:
        y.append([SECSTR.index(s) for s in l])
    return x, y


def sample_test(filename, m, init_epoch=None):
    # read data from file
    features, labels = read_data(filename)
    # preprocess data to [features, labels]
    x_in, y_in = preprocess_data(features, labels)
    print(x_in[0], y_in[0])
    m.build_train_model()
    if init_epoch:
        m.load_model_weights(init_epoch=init_epoch)
    m.fit_generator(
        generator=m.data_generator(x_in, y_in), epochs=1, steps_per_epoch=1,
        validation_data=m.data_generator(x_in, y_in), validation_steps=1,
    )
    x_len = len(x_in[-1])
    test_x = np.array(x_in[-1]).reshape(1, x_len, INPUT_SIZE)
    test_y = np.array(y_in[-1]).reshape(1, x_len, OUTPUT_SIZE)
    print(m.predict(test_x), test_y)


class Evaluate(Callback):
    def __init__(self, model, valid_x, valid_y):
        self._model = model
        self._valid_x, self._valid_y = valid_x, valid_y

    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self._model, 'predict'):
            for idx, x in enumerate(self._valid_x):
                self._model.predict(x, self._valid_y[idx])


def train(filename, m, init_epoch=None, valid_split=0.1, epochs=100):
    # read data from file
    features, labels = read_data(filename)
    # preprocess data to [features, labels]
    x_in, y_in = preprocess_data(features, labels)
    # generate train and validate data set
    split_index = int(len(x_in)*(1-valid_split))
    train_x_in, train_y_in = x_in[:split_index], y_in[:split_index]
    valid_x_in, valid_y_in = x_in[split_index+1:], y_in[split_index+1:]
    train_size = len(train_x_in)
    valid_size = len(valid_x_in)
    train_model = m.build_train_model()
    evaluation_callback = Evaluate(m, x_in[-2:], y_in[-2:])
    checkpoint_callback = ModelCheckpoint(m.checkpoint_path())
    train_generator = m.data_generator(train_x_in, train_y_in)
    valid_generator = m.data_generator(valid_x_in, valid_y_in)
    # start training
    if init_epoch:
        m.load_model_weights(init_epoch=init_epoch)
        train_model.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=train_size,
            validation_data=valid_generator,
            validation_steps=valid_size,
            callbacks=[checkpoint_callback, evaluation_callback],
            initial_epoch=init_epoch,
        )
    else:
        train_model.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=train_size,
            validation_data=valid_generator,
            validation_steps=valid_size,
            callbacks=[checkpoint_callback, evaluation_callback],
        )


def predict(in_file, m, init_epoch, out_file):
    # read data from file
    features = read_test_data(in_file)
    # preprocess data to features
    x_in, _ = preprocess_data(features)
    encoder_model, decoder_model = m.build_predict_model()
    m.load_model_weights(init_epoch=init_epoch, model=encoder_model)
    m.load_model_weights(init_epoch=init_epoch, model=decoder_model)
    y_out = []
    for x in x_in:
        y_out.extend(m.predict(x, encoder_model, decoder_model))
    with open(out_file, 'w') as f:
        f.write('class\n')
        for y in y_out:
            f.write('{}\n'.format(y))
    return y_out


def predict_ex(in_file, m, init_epoch, out_file):
    # read data from file
    features = read_test_data(in_file)
    # preprocess data to features
    x_in, _ = preprocess_data(features)
    model = m.build_predict_model()
    m.load_model_weights(init_epoch=init_epoch, model=model)
    y_out = []
    for x in x_in:
        y_out.extend(m.predict(x, model))
    with open(out_file, 'w') as f:
        f.write('class\n')
        for y in y_out:
            f.write('{}\n'.format(y))
    return y_out


if __name__ == "__main__":
    # m = BenchMarkModel('benchmark')
    # m = BidirectionModel('bidirection-2')
    # m = EncoderDecoderModel('encoder-decoder')
    # m = LatestEncoderDecoderModel('latest-encoder-decoder', maxlen=100)
    m = AttentionEncoderDecoderModel('attention-encoder-decoder', maxlen=100)
    train("ss100_train.txt", m, init_epoch=None)
    # sample_test('ss100_train.txt', m)
    # predict('ss100_test.txt', m, 100, 'encoder-decoder_out.csv')
