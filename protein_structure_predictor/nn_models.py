# coding: utf-8
import os
import numpy as np
from keras import models
from keras.layers import (
    Input, LSTM, Dense, TimeDistributed, Bidirectional, CuDNNLSTM, Embedding, RepeatVector,
    Lambda, Permute, multiply, concatenate, Masking, add, Activation, LSTMCell, RNN, CuDNNGRU,
    GRUCell, GRU,
)
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
# from keras.optimizers import Adam, SGD

from data_mixin import DataMixin
from recurrent_attention import DenseAnnotationAttention


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


class AttentionEncoderDecoderModel(EncoderDecoderModel):
    """refer to: https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
    """

    def attention_3d_block(self, inputs, single_attention_vector=True, name=''):
    	# inputs.shape = (batch_size, time_steps, input_dim)
    	input_dim = int(inputs.shape[2])
        timestep_dim = int(inputs.shape[1])
    	a = Permute((2, 1))(inputs)
    	a = Dense(timestep_dim, name='{}_attention_dense'.format(name), activation='softmax')(a)
    	if single_attention_vector:
            # 共享单一attention向量
            # 此时axis=1为input_dim维度
    	    a = Lambda(lambda x: K.mean(x, axis=1), name='{}_dim_reduction'.format(name))(a)
    	    a = RepeatVector(input_dim)(a)
    	a_probs = Permute((2, 1), name='{}_attention_vec'.format(name))(a)
    	output_attention_mul = multiply([inputs, a_probs], name='{}_attention_mul'.format(name))
        return output_attention_mul

    def build_train_model(self):
        latent_dim = 64
        target_class_number = self._output_size + 1  # 包含mark，总共4个类别
        encoder_input = Input(shape=(self._maxlen, self._input_size + 1))
        encoder = Bidirectional(CuDNNLSTM(
            latent_dim, return_sequences=True, name='encoder_lstm_1'
        ))(encoder_input)
        encoder_output = Bidirectional(CuDNNLSTM(
            latent_dim, return_sequences=True, name='encoder_lstm_2'
        ))(encoder)
        # encoder attention after LSTM
        encoder_attention = self.attention_3d_block(
            encoder_output, single_attention_vector=True, name='encoder_attention',
        )
        decoder_output = CuDNNLSTM(
            latent_dim, return_sequences=True, name='decoder_lstm_1'
        )(concatenate([encoder_input, encoder_attention]))
        decoder_attention = self.attention_3d_block(
            decoder_output, single_attention_vector=True, name='decoder_attention',
        )
        final_output = TimeDistributed(
            Dense(target_class_number, activation='softmax', name='decoder_softmax')
        )(concatenate([encoder_input, encoder_attention, decoder_output, decoder_attention]))
        model = models.Model(encoder_input, final_output)
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


class AttentionGRUModel(EncoderDecoderModel):
    """
       refer to:
         https://guillaumegenthial.github.io/sequence-to-sequence.html
         https://github.com/keras-team/keras/blob/7cc25bec619073735b72e6499028fa14e8ea99a6/examples/recurrent_attention_machine_translation.py
    """

    # def build_train_model(self):
    #     latent_dim = 64
    #     target_class_number = self._output_size + 2  # 包含mark,start_token，总共5个类别
    #     encoder_input = Input(shape=(self._maxlen, self._input_size + 1))
    #     encoder = CuDNNLSTM(
    #         latent_dim, return_sequences=True, name='encoder_lstm_1'
    #     )(encoder_input)
    #     encoder_output = CuDNNLSTM(
    #         latent_dim, return_sequences=True, name='encoder_lstm_2'
    #     )(encoder)

    #     decoder_input = Input(shape=(self._maxlen, target_class_number))
    #     # encoder attention after LSTM
    #     attention_output = self.attention_3d_block(
    #         encoder_output, single_attention_vector=True
    #     )
    #     decoder_output = CuDNNLSTM(
    #         latent_dim, return_sequences=True, name='decoder_lstm_1'
    #     )(decoder_input, constants=attention_output)

    #     final_output = TimeDistributed(
    #         Dense(target_class_number, activation='softmax', name='decoder_softmax')
    #     )(decoder_output)
    #     model = models.Model([encoder_input, decoder_input], final_output)
    #     model.compile(
    #         optimizer='adam',
    #         loss='categorical_crossentropy',
    #         metrics=['accuracy']
    #     )
    #     model.summary()
    #     self._model = model
    #     return model

    def build_train_model(self):
        latent_dim = 64
        target_class_number = self._output_size + 1  # 包含start, 总共4个类别
        x = Input(shape=(None, self._input_size))
        x_enc, h_enc_fwd_final, h_enc_bkw_final = Bidirectional(GRU(
            latent_dim, return_sequences=True, return_state=True, name='encoder_gru_1',
            dropout=0.1, recurrent_dropout=0.1,
        ))(x)

        u = TimeDistributed(Dense(latent_dim, use_bias=False))(x_enc)
        y = Input(shape=(None, target_class_number))
        initial_state_gru = Dense(latent_dim, activation='tanh')(h_enc_bkw_final)
        initial_attention_h = Lambda(lambda _x: K.zeros_like(_x)[:, 0, :])(x_enc)
        initial_state = [initial_state_gru, initial_attention_h]
        h1 = RNN(
            cell=DenseAnnotationAttention(
                cell=GRUCell(latent_dim, dropout=0.1, recurrent_dropout=0.1),
                input_mode='concatenate',
                output_mode='cell_output',
            ),
            return_sequences=True, name='decoder_gru_1',
        )(y, initial_state=initial_state, constants=[x_enc, u])
        y_pred = TimeDistributed(
            Dense(target_class_number, activation='softmax', name='decoder_softmax')
        )(concatenate([h1, y]))
        model = models.Model([x, y], y_pred)
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
        target_class_number = self._output_size + 1  # mask: 0, start: 1
        x_in = [self.one_hot(np.array(self.amido2id(x, mask=False)), self._input_size) for x in data_x]
        y_in = [to_categorical(self.sec2id(y, mask=False, start_token=True), num_classes=target_class_number) for y in data_y]
        pairs = [
            (x, y_in[i]) for i, x in enumerate(x_in)
        ]
        while True:
            np.random.shuffle(pairs)
	    for p in pairs:
		l = len(p[0])
                x = np.array(p[0]).reshape(1, l, self._input_size).astype(np.float32, copy=False)
                y = np.array(p[1]).reshape(1, l + 1, target_class_number).astype(np.float32, copy=False)
                yield [x, y[:, :-1]], y[:, 1:, :]

    def _vanilla_predict(self, x_input):
        assert(self._model)
        target_class_number = self._output_size + 1  # start: 1
        x_len = len(x_input)
        x_in = self.one_hot(np.array(self.amido2id(x_input, mask=False)), self._input_size)
        padding_x = x_in.reshape(1, x_len, self._input_size).astype(np.float32, copy=False)
        # encode the input as state vectors.
        y_in = [self.START_TOKEN-1]
        while len(y_in) < x_len + 1:
            padding_y = np.array(
                to_categorical(y_in, num_classes=target_class_number)
            ).reshape(1, len(y_in), target_class_number).astype(np.float32, copy=False)
            y_out = self._model.predict([padding_x, padding_y])
            # y_in = np.concatenate((np.array([self.START_TOKEN-1]), np.argmax(y_out[0, :, :], axis=-1)), axis=None)
            idx = np.argmax(y_out[0, -1, :])  # 忽略mask和start
            y_in.append(idx)
        return y_in[1:]

    def _beam_search(self, x_input, topk=3):
        target_class_number = self._output_size + 1
        scores = [0] * topk
        yid = np.array([[self.START_TOKEN-1]] * topk)
        xid = np.array([self.one_hot(np.array(self.amido2id(x_input, mask=False)), self._input_size)] * topk)
        xid = xid.reshape(topk, len(x_input), self._input_size)
        for i in range(len(x_input)):
            y = np.array(
                to_categorical(yid, num_classes=target_class_number)
            ).reshape(topk, yid.shape[1], target_class_number).astype(np.float32, copy=False)
            t = self._model.predict([xid, y])
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
        return yid[np.argmax(scores)][1:] # remove start token

    def predict(self, x_input, y_output=None):
        greedy_y_pred = self._vanilla_predict(x_input)
        beam_y_pred = self._beam_search(x_input)
        print 'input x:'
        print ''.join(x_input)
        print 'greedy predt y:'
        print ''.join(self.id2sec(greedy_y_pred, mask=False, start_token=True))
        print 'beam search predt y:'
        print ''.join(self.id2sec(beam_y_pred, mask=False, start_token=True))
        if y_output:
            print 'truth y:'
            print ''.join(y_output)
        return np.array(greedy_y_pred) - 1
