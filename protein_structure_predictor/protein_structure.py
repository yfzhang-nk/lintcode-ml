# coding: utf-8
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

from nn_models import *


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


class Evaluate(Callback):
    def __init__(self, model, valid_x, valid_y):
        self._model = model
        self._valid_x, self._valid_y = valid_x, valid_y

    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self._model, 'predict'):
            if hasattr(self._model, 'load_weight_when_evaluation'):
                self._model.load_weight_when_evaluation(epoch)
            for idx, x in enumerate(self._valid_x):
                self._model.predict(x, self._valid_y[idx])


def train(filename, m, init_epoch=None, valid_split=0.01, epochs=100):
    # read data from file
    x_in, y_in = read_data(filename)
    # generate train and validate data set
    split_index = int(len(x_in)*(1-valid_split))
    train_x_in, train_y_in = x_in[:split_index], y_in[:split_index]
    valid_x_in, valid_y_in = x_in[split_index+1:], y_in[split_index+1:]
    train_size = len(train_x_in)
    valid_size = len(valid_x_in)
    train_model = m.build_train_model()
    evaluation_callback = Evaluate(m, x_in[-2:], y_in[-2:])
    checkpoint_callback = ModelCheckpoint(m.checkpoint_path())
    tensorboard_callback = TensorBoard(log_dir='./logs')
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
            callbacks=[checkpoint_callback, evaluation_callback, tensorboard_callback],
            initial_epoch=init_epoch,
        )
    else:
        train_model.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=train_size,
            validation_data=valid_generator,
            validation_steps=valid_size,
            callbacks=[checkpoint_callback, evaluation_callback, tensorboard_callback],
        )


def predict(in_file, m, init_epoch, out_file):
    # read data from file
    x_in = read_test_data(in_file)
    encoder_model, decoder_model = m.build_predict_model()
    m.load_model_weights(init_epoch=init_epoch, model=encoder_model)
    m.load_model_weights(init_epoch=init_epoch, model=decoder_model)
    y_out = []
    for x in x_in:
        y_out.extend(m.predict(x))
    with open(out_file, 'w') as f:
        f.write('class\n')
        for y in y_out:
            f.write('{}\n'.format(y))
    return y_out


def predict_ex(in_file, m, init_epoch, out_file):
    # read data from file
    x_in = read_test_data(in_file)
    model = m.build_predict_model()
    m.load_model_weights(init_epoch=init_epoch, model=model)
    y_out = []
    for x in x_in:
        y_out.extend(m.predict(x))
    with open(out_file, 'w') as f:
        f.write('class\n')
        for y in y_out:
            f.write('{}\n'.format(y))
    return y_out


if __name__ == "__main__":
    # m = BenchMarkModel('benchmark', input_size=26, output_size=3)
    # predict_ex('ss100_test.txt', m, 30, 'benchmark_out.csv')

    # m = BidirectionModel('bidirection', input_size=26, output_size=3)
    # predict_ex('ss100_test.txt', m, 20, 'bidirection_out.csv')

    # m = Seq2SeqModel('seq2seq', input_size=26, output_size=3)
    # predict('ss100_test.txt', m, 100, 'encoder-decoder_out.csv')

    # m = EncoderDecoderModel('encoder-decoder', input_size=26, output_size=3, maxlen=100)
    # predict_ex('ss100_test.txt', m, 99, 'encoder-decoder_out.csv')

    m = AttentionEncoderDecoderModel('more-attention-encoder-decoder', input_size=26, output_size=3, maxlen=100)
    predict_ex('ss100_test.txt', m, 18, 'more-attention-encoder-decoder_out.csv')

    # m = AttentionGRUModel('attention-gru', input_size=26, output_size=3, maxlen=100)
    # predict_ex('ss100_test.txt', m, 62, 'attention-gru_out.csv')

    # train("ss100_train.txt", m, init_epoch=None)
    # sample_test('ss100_train.txt', m)
    # predict_ex('ss100_test.txt', m, 30, 'benchmark_out.csv')
