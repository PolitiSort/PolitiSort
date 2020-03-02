import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Lambda, concatenate
import keras.backend as K

class PolitiNet(object):
    def __init__(self):
        self.model = self.__build()

    @staticmethod
    def __build():
        inp_bio = Input(shape=(None, None)) 
        inp_exp_bio = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp)

        inp_status = Input(shape=(None, None)) 
        inp_exp_status = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp)

        net1 = LSTM(50, return_sequences=True)(concatenate([inp_exp_bio, inp_exp_status]))
        net1 = LSTM(50, return_sequences=True)(net1)

        net2 = LSTM(32, return_sequences=True)(inp_exp)
        net2 = LSTM(32)(net2)

        net3 = Dense(128, activation="relu")(net2)
        net3 = Dense(2, activation="softmax")(net3)

        model = Model(inputs=[inp_exp_bio, inp_exp_status], outputs=[net3]) 
        return model


