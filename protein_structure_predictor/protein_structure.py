import numpy as np
from keras import models
from keras.layers import LSTM, Dense, TimeDistributed
from keras.optimizers import Adam

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
AMIDOGEN_MAPPING = {}
AMIDOGEN_SIZE = len(AMIDOGEN)
for idx, g in enumerate(AMIDOGEN):
    AMIDOGEN_MAPPING[g] = SCALE_MAX / AMIDOGEN_SIZE * (idx + 1)

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
        x.append(np.array([AMIDOGEN_MAPPING.get(s, 0.0) for s in f]))
    label_size = len(SECSTR)
    for l in labels:
        embedding = np.zeros((label_size, len(l)), dtype=np.int8)
        for idx, s in enumerate(l):
            embedding[SECSTR_MAPPING[s], idx] = 1
        y.append(embedding)
    return x, y


def build_model():
    model = models.Sequential()
    model.add(
        LSTM(
            32, return_sequences=True,
            input_shape=(None, 1),
            # dropout=0.5, recurrent_dropout=0.5,
            # activity_regularizer=keras.regularizers.l1(0.2)
        )
    )
    model.add(
        LSTM(
            16, return_sequences=True,
            # dropout=0.5, recurrent_dropout=0.5,
            # activity_regularizer=keras.regularizers.l1(0.2)
        )
    )
    model.add(TimeDistributed(Dense(3, activation='softmax')))
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())
    return model



def train_generator(x, y, split=0.1):
    split_index = int(len(x)*(1-split))
    train_x, train_y = x[:split_index], y[:split_index]
    return data_generator(train_x, train_y)


def valid_generator(x, y, split=0.1):
    split_index = int(len(x)*split)
    valid_x, valid_y = x[split_index+1:], y[split_index+1:]
    return data_generator(valid_x, valid_y)


def data_generator(data_x, data_y):
    pairs = [(x, data_y[i]) for i, x in enumerate(data_x)]
    np.random.shuffle(pairs)
    for p in pairs:
        x_len = len(p[0])
        x = np.array(p[0]).reshape(1, x_len, INPUT_SIZE)
        y = np.array(p[1]).reshape(1, x_len, OUTPUT_SIZE)
        yield (x, y)
    yield None


def train(filename):
    features, labels = read_data(filename)
    x_in, y_in = preprocess_data(features, labels)
    m = build_model()
    m.fit_generator(
        generator=train_generator(x_in, y_in), epochs=100, steps_per_epoch=len(x_in),
        validation_data=valid_generator(x_in, y_in), validation_steps=10,
    )

if __name__ == "__main__":
    train("ss100_train.txt")
