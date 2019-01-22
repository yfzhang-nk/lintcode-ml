# coding: utf-8
import numpy as np
from keras.callbacks import ModelCheckpoint

from nn_models import BenchMarkModel, BidirectionModel

SCALE_MAX = 1.0
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
    'V',
    'X',
    'Y',
    'Z',
]
AMIDOGEN_SIZE = len(AMIDOGEN)
AMIDOGEN_MAPPING = dict(zip(AMIDOGEN, [idx + 1 for idx in range(AMIDOGEN_SIZE)]))

SECSTR = ['H', 'E', 'C']
SECSTR_MAPPING = {
    'H': 0,
    'E': 1,
    'C': 2,
}

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


def preprocess_data(features, labels):
    x = []
    y = []
    for f in features:
        x.append([AMIDOGEN_MAPPING.get(s, 0) for s in f])
    for l in labels:
        y.append([SECSTR_MAPPING[s] for s in l])
    return x, y


def sample_test(filename, m, init_epoch=None):
    # read data from file
    features, labels = read_data(filename)
    # preprocess data to [features, labels]
    x_in, y_in = preprocess_data(features, labels)
    print(x_in[0], y_in[0])
    train_model = m.build_train_model()
    if init_epoch:
        train_model.load_weights(init_epoch=init_epoch)
    m.fit_generator(
        generator=m.data_generator(x_in, y_in), epochs=1, steps_per_epoch=1,
        validation_data=m.data_generator(x_in, y_in), validation_steps=1,
    )
    x_len = len(x_in[-1])
    test_x = np.array(x_in[-1]).reshape(1, x_len, INPUT_SIZE)
    test_y = np.array(y_in[-1]).reshape(1, x_len, OUTPUT_SIZE)
    print(m.predict(test_x), test_y)


def train(filename, m, init_epoch=None, valid_split=0.1):
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
    checkpoint_callback = ModelCheckpoint(m.checkpoint_path())
    train_generator = m.data_generator(train_x_in, train_y_in)
    valid_generator = m.data_generator(valid_x_in, valid_y_in)
    # start training
    if init_epoch:
        train_model.load_model_weights(init_epoch=init_epoch)
        train_model.fit_generator(
            generator=train_generator,
            epochs=100,
            steps_per_epoch=train_size,
            validation_data=valid_generator,
            validation_steps=valid_size,
            callbacks=[checkpoint_callback],
            initial_epoch=init_epoch,
        )
    else:
        train_model.fit_generator(
            generator=train_generator,
            epochs=100,
            steps_per_epoch=train_size,
            validation_data=valid_generator,
            validation_steps=valid_size,
            callbacks=[checkpoint_callback],
        )


if __name__ == "__main__":
    # m = BenchMarkModel('benchmark')
    m = BidirectionModel('bidirection')
    train(
        "ss100_train.txt", m,
    )
    # sample_test('ss100_train.txt', m)
