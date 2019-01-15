# coding: utf-8
import os
import numpy as np
from keras import models
from keras.layers import LSTM, Dense, TimeDistributed
from keras.callbacks import ModelCheckpoint
# from keras.optimizers import Adam, SGD

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
]
AMIDOGEN_SIZE = len(AMIDOGEN)
AMIDOGEN_MAPPING = dict(zip(AMIDOGEN, [SCALE_MAX / AMIDOGEN_SIZE * idx for idx in range(1, AMIDOGEN_SIZE+1)]))

SECSTR = ['H', 'E', 'C']
SECSTR_MAPPING = {
    'H': 0,
    'E': 1,
    'C': 2,
}

INPUT_SIZE = 1
OUTPUT_SIZE = 3
CHECKPOINT_DIR = './model_checkpoints'
CHECKPOINT_FILE = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

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
        x.append(np.array([AMIDOGEN_MAPPING.get(s, 0.0) for s in f]))
    label_size = len(SECSTR)
    for l in labels:
        embedding = np.zeros((label_size, len(l)), dtype=np.int8)
        for idx, s in enumerate(l):
            embedding[SECSTR_MAPPING[s], idx] = 1.0
        y.append(embedding)
    return x, y


def build_model():
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
    model.add(TimeDistributed(Dense(OUTPUT_SIZE, activation='softmax')))
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())
    return model


def data_generator(data_x, data_y):
    pairs = [(x, data_y[i]) for i, x in enumerate(data_x)]
    while True:
        np.random.shuffle(pairs)
        for p in pairs:
            x_len = len(p[0])
            x = np.array(p[0]).reshape(1, x_len, INPUT_SIZE)
            y = np.array(p[1]).reshape(1, x_len, OUTPUT_SIZE)
            yield (x, y)
    yield None


def sample_test(filename):
    # read data from file
    features, labels = read_data(filename)
    # preprocess data to [features, labels]
    x_in, y_in = preprocess_data(features, labels)
    print(x_in[0], y_in[0])
    m = build_model()
    m.fit_generator(
        generator=data_generator(x_in, y_in), epochs=1, steps_per_epoch=1,
        validation_data=data_generator(x_in, y_in), validation_steps=1,
    )
    x_len = len(x_in[-1])
    test_x = np.array(x_in[-1]).reshape(1, x_len, INPUT_SIZE)
    test_y = np.array(y_in[-1]).reshape(1, x_len, OUTPUT_SIZE)
    print(m.predict(test_x), test_y)


def train(filename, valid_split=0.1):
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
    # build network model
    m = build_model()
    checkpoint_callback = ModelCheckpoint(CHECKPOINT_PATH)
    # start training
    m.fit_generator(
        generator=data_generator(train_x_in, train_y_in), epochs=20, steps_per_epoch=train_size,
        validation_data=data_generator(valid_x_in, valid_y_in), validation_steps=valid_size,
        callbacks=[checkpoint_callback]
    )


if __name__ == "__main__":
    train("ss100_train.txt")
    # sample_test('ss100_train.txt')
