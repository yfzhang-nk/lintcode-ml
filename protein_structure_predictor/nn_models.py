# coding: utf-8
import os
import numpy as np
from keras import models
from keras.layers import (
    Input, LSTM, Dense, TimeDistributed, Bidirectional, CuDNNLSTM, Embedding, RepeatVector,
    Lambda, Permute, multiply, concatenate, Masking, add, Activation,
)
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
# from keras.optimizers import Adam, SGD

from data_mixin import DataMixin


class BaseModel(DataMixin):
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

    def load_model_weights(self, init_epoch=None, model=None):
        if init_epoch:
            weights_file = os.path.join(
                self._model_dir,
                self._file_pattern.format(epoch=init_epoch)
            )
            if model:
                model.load_weights(weights_file, by_name=True)
            elif self._model:
                self._model.load_weights(weights_file)
            print '===weights [{}] loaded==='.format(weights_file)

    def data_generator(self, data_x, data_y):
        pairs = [
            ([self.one_hot(np.array(xi), self._input_size) for xi in self.amido2id(x, mask=False)], to_categorical(
                self.sec2id(data_y[i], mask=False), num_classes=self._output_size
            )) for i, x in enumerate(data_x)
        ]
        while True:
            np.random.shuffle(pairs)
            for p in pairs:
                x_len = len(p[0])
                x = np.array(p[0]).reshape(1, x_len, self._input_size)
                y = np.array(p[1]).reshape(1, x_len, self._output_size)
                y = y.astype(np.float32, copy=False)
                yield (x, y)

    def predict(self, x_input, y_output=None):
        assert(self._model)
        x_in = self.one_hot(np.array(self.amido2id(x_input, mask=False)), self._input_size)
        x_in = np.array(x_in).reshape(1, len(x_input), self._input_size)
        y_out = self._model.predict(x_in)
        y_out = np.argmax(y_out[0, :, :], axis=-1)
        print 'input x:', ''.join(x_input)
        print 'predt y:', ''.join(self.id2sec(y_out, mask=False))
        if y_output:
            print 'truth y:', ''.join(y_output)
        return y_out


class BenchMarkModel(BaseModel):
    def _build_model(self):
        model = models.Sequential()
        model.add(
            CuDNNLSTM(
                32, return_sequences=True,
                input_shape=(None, 26),
            )
        )
        model.add(
            CuDNNLSTM(
                32, return_sequences=True,
            )
        )
        model.add(
            CuDNNLSTM(
                16, return_sequences=True,
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


class BidirectionModel(BaseModel):

    def _build_model(self):
        model = models.Sequential()
        model.add(
            Bidirectional(CuDNNLSTM(
                16, return_sequences=True,
            ), input_shape=(None, 26))
        )
        model.add(
            Bidirectional(CuDNNLSTM(
                16, return_sequences=True,
            ))
        )
        model.add(
            Bidirectional(CuDNNLSTM(
                8, return_sequences=True,
            ))
        )
        model.add(Dense(64, activation='relu'))
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


class Seq2SeqModel(BaseModel):
    """
    refer to:
       https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
    """
    def build_train_model(self):
        latent_dim = 64
        target_class_number = self._output_size + 1  # 包含start_token，总共4个类别
        encoder_input = Input(shape=(None, 26))
        encoder = CuDNNLSTM(
            latent_dim, return_sequences=True, name='encoder_lstm_1'
        )(encoder_input)
        encoder_outputs, state_h, state_c = CuDNNLSTM(
            latent_dim, return_state=True, name='encoder_lstm_2'
        )(encoder)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        decoder_input = Input(shape=(None, target_class_number))
        decoder = CuDNNLSTM(
            latent_dim, return_sequences=True, name='decoder_lstm_1'
        )(decoder_input, initial_state=encoder_states)
        decoder_output, _, _ = CuDNNLSTM(
            latent_dim, return_sequences=True, return_state=True, name='decoder_lstm_2'
        )(decoder, initial_state=encoder_states)
        decoder_output = Dense(target_class_number, activation='softmax', name='decoder_softmax')(decoder_output)
        model = models.Model([encoder_input, decoder_input], decoder_output)
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        self._model = model
        return model

    def build_predict_model(self):
        latent_dim = 64
        target_class_number = self._output_size + 1  # 包含start_token，总共4个类别
        encoder_input = Input(shape=(None, 26))
        encoder = CuDNNLSTM(
            latent_dim, return_sequences=True, name='encoder_lstm_1'
        )(encoder_input)
        encoder_outputs, state_h, state_c = CuDNNLSTM(
            latent_dim, return_state=True, name='encoder_lstm_2'
        )(encoder)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        encoder_model = models.Model(encoder_input, encoder_states)

        decoder_input = Input(shape=(None, target_class_number))
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder = CuDNNLSTM(
            latent_dim, return_sequences=True, name='decoder_lstm_1'
        )(decoder_input, initial_state=decoder_states_inputs)
        decoder_output, state_h, state_c = CuDNNLSTM(
            latent_dim, return_sequences=True, return_state=True, name='decoder_lstm_2'
        )(decoder, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_output = Dense(target_class_number, activation='softmax', name='decoder_softmax')(decoder_output)
        decoder_model = models.Model(
            [decoder_input] + decoder_states_inputs,
            [decoder_output] + decoder_states)
        self._encoder_model, self._decoder_model = encoder_model, decoder_model
        return encoder_model, decoder_model

    def data_generator(self, data_x, data_y):
        pairs = [
            ([self.one_hot(np.array(xi), self._input_size) for xi in self.amido2id(x, mask=False, reverse=True)], to_categorical(
                self.sec2id(data_y[i], mask=False, start_token=True), num_classes=self._output_size + 1
            )) for i, x in enumerate(data_x)
        ]
        while True:
            np.random.shuffle(pairs)
            for p in pairs:
                x_len = len(p[0])
                x = np.array(p[0]).reshape(1, x_len, self._input_size)
                y = np.array(p[1]).reshape(1, x_len + 1, self._output_size + 1)
                y = y.astype(np.float32, copy=False)
                yield [x, y[:, :-1, :]], y[:, 1:, :]

    def load_weight_when_evaluation(self, epoch=None):
        self.build_predict_model()
        self.load_model_weights(init_epoch=epoch, model=self._encoder_model)
        self.load_model_weights(init_epoch=epoch, model=self._decoder_model)

    def _beam_search(self, x_input, topk=3):
        scores = [0] * topk
        yid = np.array([[self.START_TOKEN-1]] * topk)
        xid = np.array([self.one_hot(np.array(self.amido2id(x_input, mask=False, reverse=True)), self._input_size)] * topk)
        xid = xid.reshape(topk, len(x_input), self._input_size)
        # encode the input as state vectors.
        states_value = self._encoder_model.predict(xid)
        for i in range(len(x_input)):
            y = to_categorical(yid, num_classes=self._output_size + 1)
            y = y.reshape(topk, yid.shape[1], self._output_size + 1)
            y = y.astype(np.float32, copy=False)
            t, h, c = self._decoder_model.predict([y] + states_value)
            proba = t[:, i, 1:]  # ignore start token
            log_proba = np.log(proba + 1e-8)
            arg_topk = log_proba.argsort(axis=1)[:, -topk:]
            _yid = []  # 暂存的候选目标序列
            _scores = []  # 暂存的候选目标序列得分
            if i == 0:
                for j in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[0][j]])  # topk组数据相同，取第1组数据即可
                    _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
            else:
                for j in range(len(xid)):
                    for k in range(topk):
                        _yid.append(list(yid[j]) + [arg_topk[j][k]])
                        _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
                _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
                _yid = [_yid[k] for k in _arg_topk]
                _scores = [_scores[k] for k in _arg_topk]
            yid = np.array([_yid[k] for k in range(len(xid))])
            scores = [_scores[k] for k in range(len(xid))]
            # update states
            states_value = [h, c]
        return yid[np.argmax(scores)][1:] # remove start token

    def _vanilla_predict(self, x_input):
        assert(self._model)
        x_in = self.one_hot(np.array(self.amido2id(x_input, mask=False, reverse=True)), self._input_size)
        x_in = np.array(x_in).reshape(1, len(x_input), self._input_size)
        y_in = [self.START_TOKEN]
        # encode the input as state vectors.
        states_value = self._encoder_model.predict(x_in)
        while len(y_in) <= len(x_input):
            y = to_categorical(y_in, num_classes=self._output_size + 1)
            y = np.array(y).reshape(1, len(y_in), self._output_size + 1)
            y = y.astype(np.float32, copy=False)
            t, h, c = self._decoder_model.predict([y] + states_value)
            y_in = np.concatenate((np.array([self.START_TOKEN]), np.argmax(t[0, :, :], axis=-1)), axis=None)
            # y_in.append(np.argmax(t[0, -1, :], axis=-1))
            # update states
            states_value = [h, c]
        y_out = y_in[1:]  # remove start token
        return y_out

    def predict(self, x_input, y_output=None):
        y_out = self._vanilla_predict(x_input)
        # y_out = self._beam_search(x_input)
        print 'input x:', ''.join(x_input)
        print 'predt y:', ''.join(self.id2sec(y_out, mask=False, start_token=True))
        if y_output:
            print 'truth y:', ''.join(y_output)
        return y_out


class EncoderDecoderModel(BaseModel):
    """
    refer to:
        https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
    """
    def __init__(self, *args, **kwargs):
        self._maxlen = kwargs.pop('maxlen')
        super(EncoderDecoderModel, self).__init__(*args, **kwargs)

    def _build_model(self):
        latent_dim = 32
        target_class_number = self._output_size + 1
        encoder_input = Input(shape=(self._maxlen, self._input_size + 1))
        encoder = Bidirectional(CuDNNLSTM(
            latent_dim / 2, return_sequences=True,
        ))(encoder_input)
        encoder = Bidirectional(CuDNNLSTM(
            latent_dim / 2,
        ))(encoder)
        encoder = RepeatVector(self._maxlen)(encoder)
        decoder = CuDNNLSTM(
            latent_dim, return_sequences=True,
        )(encoder)
        decoder = CuDNNLSTM(
            latent_dim, return_sequences=True,
        )(decoder)
        decoder_output = TimeDistributed(Dense(target_class_number, activation='softmax'))(decoder)
        model = models.Model(encoder_input, decoder_output)
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
        source_dim = self._input_size + 1
        target_dim = self._output_size + 1
        x_in = [self.amido2id(x, mask=True) for x in data_x]
        x_in = pad_sequences(x_in, maxlen=self._maxlen, padding='post', value=0)
        x_in = [self.one_hot(x, source_dim) for x in x_in]
        x_in = np.array(x_in).astype(np.float32, copy=False)
        y_in = [self.sec2id(y, mask=True, start_token=False) for y in data_y]
        y_in = pad_sequences(y_in, maxlen=self._maxlen, padding='post', value=0)
        y_in = [to_categorical(y, num_classes=target_dim) for y in y_in]
        y_in = np.array(y_in).astype(np.float32, copy=False)
        pairs = [
            (x, y_in[i]) for i, x in enumerate(x_in)
        ]
        while True:
            np.random.shuffle(pairs)
            for p in pairs:
                x = np.array(p[0]).reshape(1, self._maxlen, source_dim)
                y = np.array(p[1]).reshape(1, self._maxlen, target_dim)
                yield (x, y)

    def predict(self, x_input, y_output=None):
        x_in = [self.amido2id(x_input, mask=True)]
        x_in = pad_sequences(x_in, maxlen=self._maxlen, padding='post', value=0)
        x_in = self.one_hot(x_in, self._input_size+1)
        x_in = x_in.astype(np.float32, copy=False)
        y_out = self._model.predict(x_in)
        y_out = np.argmax(y_out[0, :len(x_input), :], axis=-1)
        print 'input x:', ''.join(x_input)
        print 'predt y:', ''.join(self.id2sec(y_out, mask=True, start_token=False))
        if y_output:
            print 'truth y:', ''.join(y_output)
        return np.array(y_out) - 1


class AttentionEncoderDecoderModel(BaseModel):
    """refer to: https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
    """
    def __init__(self, *args, **kwargs):
        self._maxlen = kwargs.pop('maxlen')
        super(AttentionEncoderDecoderModel, self).__init__(*args, **kwargs)

    def attention_3d_block(self, q, v, single_attention_vector=True):
    	# q.shape = (batch_size, time_steps, input_dim)
    	# input_dim = int(v.shape[2])
        timestep_dim = int(v.shape[1])
    	a = Permute((2, 1))(q)
        b = Permute((2, 1))(v)
        # 把查询q在维度上和v的timestep对齐
    	a = Dense(timestep_dim, name='attention_dense_q')(a)
        b = Dense(timestep_dim, name='attention_dense_v')(b)
        m = Activation('softmax')(add([a, b]))
    	# if single_attention_vector:
        #     # 共享单一attention向量
        #     # 此时axis=1为input_dim维度
    	#     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    	#     a = RepeatVector(input_dim)(a)
    	a_probs = Permute((2, 1), name='attention_vec')(m)
    	output_attention_mul = multiply([v, a_probs], name='attention_mul')
        return output_attention_mul

    def build_train_model(self):
        latent_dim = 32
        target_class_number = self._output_size + 2  # 包含mark,start_token，总共5个类别
        encoder_input = Input(shape=(self._maxlen, ))
        encoder = Bidirectional(CuDNNLSTM(
            latent_dim / 2, return_sequences=True, name='encoder_lstm_1'
        ))(encoder_input)
        encoder_output = Bidirectional(CuDNNLSTM(
            latent_dim / 2, return_sequences=True, name='encoder_lstm_2'
        ))(encoder)

        decoder_input = Input(shape=(self._maxlen, ))
        decoder = CuDNNLSTM(
            latent_dim, return_sequences=True, name='decoder_lstm_1'
        )(decoder_input)
        decoder_output = CuDNNLSTM(
            latent_dim, return_sequences=True, name='decoder_lstm_2'
        )(decoder)

        # encoder attention after LSTM
        attention_output = self.attention_3d_block(
            q=decoder_output, v=encoder_output, single_attention_vector=False
        )
        merged_output = concatenate([attention_output, decoder_output])

        final_output = TimeDistributed(
            Dense(target_class_number, activation='softmax', name='decoder_softmax')
        )(merged_output)
        model = models.Model([encoder_input, decoder_input], final_output)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        self._model = model
        return model

    def build_predict_model(self):
        return self.build_train_model()

    def data_generator(self, data_x, data_y):
        target_class_number = self._output_size + 2  # mask: 0, start: 1
        x_in = [self.amido2id(x) for x in data_x]
        y_in = [self.sec2id(y, start_token=True) for y in data_y]
        y_out = [to_categorical(y, num_classes=target_class_number) for y in y_in]
        x_in = pad_sequences(x_in, maxlen=self._maxlen, padding='post', value=0) # padding x_in
        x_in = [self.one_hot(x, self._input_size + 1) for x in x_in]
        y_in = [np.concatenate((y, [0] * (self._maxlen + 1 - len(y))), axis=0) for y in y_in] # padding y_in
        y_out = pad_sequences(y_out, maxlen=self._maxlen + 1, padding='post', value=0.0) # padding y_out
        y_out = y_out.astype(np.float32, copy=False)
        pairs = [
            (x, y_in[i], y_out[i]) for i, x in enumerate(x_in)
        ]
        while True:
            np.random.shuffle(pairs)
            for p in pairs:
                x = np.array(p[0]).reshape(1, self._maxlen, self._input_size + 1)
                y_i = np.array(p[1]).reshape(1, self._maxlen + 1)
                y_o = np.array(p[2]).reshape(1, self._maxlen + 1, target_class_number)
                yield [x, y_i[:, :-1]], y_o[:, 1:, :]

    def predict(self, x_input, y_output=None):
        assert(self._model)
        x_len = len(x_input)
        x_in = self.amido2id(x_input)
        padding_x = pad_sequences([x_in], maxlen=self._maxlen, padding='post', value=0).reshape(1, self._maxlen)
        # encode the input as state vectors.
        y_in = [self.START_TOKEN]
        while len(y_in) < x_len + 1:
            padding_y = np.array(y_in + [0] * (self._maxlen - len(y_in))).reshape(1, self._maxlen)
            y_out = self._model.predict([padding_x, padding_y])
            idx = np.argmax(y_out[0, len(y_in) - 1, 2:])  # 忽略mask和start
            y_in.append(idx)
        print 'input x:', ''.join(self.id2amido(x_in))
        print 'predt y:', ''.join(self.id2sec(y_in[1:], start_token=True))
        if y_output:
            print 'truth y:', ''.join(y_output)
            assert(len(y_in) - 1 == len(y_output))
        return y_in[1:]  # remove start_token
