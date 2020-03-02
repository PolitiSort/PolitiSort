import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Masking, LSTM, Lambda, Dropout, concatenate
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

class PolitiNet(object):
    def __init__(self, maxDesc=15, maxStatus=15, lr=3e-2):
        self.__maxDesc = maxDesc
        self.__maxStatus = maxStatus
        self.model = self.__build()
        self.model.compile(Adam(lr=lr), loss="binary_crossentropy", metrics=["acc"])

    def __build(self):
        # inp_bio = Input(shape=(self.__maxDesc,)) 

        inp_status = Input(shape=(self.__maxStatus,)) 

        inp_masked = Masking(mask_value=0.0)(inp_status)

        # inp_exp_bio = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp_bio)

        inp_exp_status = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp_masked)

        net1 = LSTM(64, return_sequences=True, activation="tanh")(inp_exp_status)
        net1 = LSTM(32, return_sequences=True, activation="tanh")(net1)
        net1 = LSTM(16, activation="tanh")(net1)

        net3 = Dense(128, activation="tanh")(net1)
        net3 = Dense(32, activation="tanh")(net1)
        net3 = Dense(32, activation="tanh")(net3)
        net3 = Dense(2, activation="softmax")(net3)

        model = Model(inputs=[inp_status], outputs=[net3]) 
        return model

    def fit(self, handler, epochs=10, batch_size=32, val_split=0.05):
        input_data = handler()
        self.model.fit(x=[pad_sequences(input_data["status"], maxlen=self.__maxStatus)], y=[input_data["isDem"]], epochs=epochs, batch_size=batch_size, validation_split=val_split, shuffle=True)

