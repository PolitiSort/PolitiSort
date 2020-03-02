import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Dropout, concatenate
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

class PolitiNet(object):
    def __init__(self, maxDesc=15, maxStatus=15):
        self.__maxDesc = maxDesc
        self.__maxStatus = maxStatus
        self.model = self.__build()
        self.model.compile("adam", loss="binary_crossentropy", metrics=["acc"])

    def __build(self):
        # inp_bio = Input(shape=(self.__maxDesc,)) 

        inp_status = Input(shape=(self.__maxStatus,)) 

        # inp_exp_bio = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp_bio)

        inp_exp_status = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp_status)

        net1 = LSTM(50, return_sequences=True)(inp_exp_status)
        net1 = LSTM(50, return_sequences=True)(net1)

        net2 = LSTM(32, return_sequences=True)(net1)
        net2 = LSTM(32)(net2)

        net3 = Dense(32, activation="relu")(net2)
        net3 = Dense(2, activation="softmax")(net3)

        model = Model(inputs=[inp_status], outputs=[net3]) 
        return model

    def fit(self, handler, epochs=10, batch_size=32, val_split=0.05):
        input_data = handler()
        self.model.fit(x=[pad_sequences(input_data["status"], maxlen=self.__maxStatus)], y=[input_data["isDem"]], epochs=epochs, batch_size=batch_size, validation_split=val_split, shuffle=True)

