import keras.backend as K
import keras
import numpy as np
from keras.layers import LSTM, Input

AMINO = dict(zip(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'], \
                    [ x/10 for x in range(-10, 11) if x != 0]))
SECSTR = dict(zip(['H', 'E', 'C'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
def read_data(filename):
    feature, label = [], []
    with open(filename, "r") as f:
        lines = f.readlines()
        for idx, val in enumerate(lines):
            val = val.strip()
            if idx % 4 == 1:
                feature.append(val)
            elif idx % 4 == 3:
                label.append(val)
    return feature, label

def build_feature_vector(feature):
    vector = []
    for amino in feature:
        vector.append([AMINO.get(x, 0) for x in amino])
    return np.array(vector)

def build_label_vetor(label):    
    vector = []
    for secstr in label:
        vector.append([SECSTR.get(x, [0, 0, 0]).copy() for x in secstr])
    return np.array(vector)

#def build_model():
#    I = Input(shape=(None, 1)) # unknown timespan, fixed feature size
#    model = keras.models.Sequential()
#    model.add(
#        keras.layers.recurrent.LSTM(
#            32, return_sequences=True,
#            # dropout=0.5, recurrent_dropout=0.5,
#            # activity_regularizer=keras.regularizers.l1(0.2)
#        )
#    )
#    model.add(
#        keras.layers.recurrent.LSTM(
#            16, return_sequences=True,
#            # dropout=0.5, recurrent_dropout=0.5,
#            # activity_regularizer=keras.regularizers.l1(0.2)
#        )
#    )
#    model.add(keras.layers.core.Dense(3, activation='softmax'))
## sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    f = K.function(inputs=[I], outputs=[model(I)])
#    f.compile(
#        optimizer=adam,
#        loss='categorical_crossentropy',
#        metrics=['accuracy']
#    )
#    model.fit(
#        data, labels, epochs=epochs,
#        batch_size=batch_size, shuffle=True, validation_split=0.2
#    )

def train(filename):
    feature, label = read_data(filename)
    print(label[0])
    feature = build_feature_vector(feature)
    label = build_label_vetor(label)
    print(feature[0])
    print(label[0])

if __name__ == "__main__":
    train("ss100_train.txt")
