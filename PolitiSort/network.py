import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Masking, LSTM, Lambda, Dropout, Conv1D, BatchNormalization, ZeroPadding1D, Flatten, UpSampling1D, Reshape
# Dependencies used in the version of Politisort that takes Bios into account
# from keras.layers import concatenate, LeakyReLU
from keras.layers import UpSampling2D, Conv2D, LeakyReLU, Activation


import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model


class PolitiNet(object):
    def __init__(self, maxdesc=15, maxstatus=15, lr=5e-3):
        self.__maxDesc = maxdesc
        self.__maxStatus = maxstatus
        self.modelGenerator = None
        self.modelDiscriminator = None
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
##############################################################
#
#
#
##############################################################
class PolitiGen(object):
    def __init__(self):
        self.model = self.build_combined()

    def __build_discriminator(self):
        """
        Put together a CNN that will return a single confidence output.

        returns: the model object
        """
        model = Sequential()
        #model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=(1,), padding="same"))
        model.add(Dense(32), input_shape=(1,))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(64))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(128))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(256))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def __build_generator(self):
        """
        Put together a model that takes in one-dimensional noise and outputs one-dimensional
        data representing words with each number representing an index in our corpus.

        returns: the model object
        """
        noise_shape = (1,)

        model = Sequential()

        model.add(Dense(64, activation="relu", input_shape=noise_shape))
        model.add(Dense(128))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        model.add(Activation("tanh"))
        return model

    def build_combined(self):
        """
        Puts together a model that combines the discriminator and generator models.

        returns: the generator, discriminator, and combined model objects
        """

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        discriminator = self.__build_discriminator()
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])

        # Build and compile the generator
        generator = self.__build_generator()

        # The generator takes noise as input and generates words
        noise = Input(shape=(1,))
        word = generator(noise)

        # For the combined model we will only train the generator
        discriminator.trainable = False

        # The discriminator takes generated words as input and determines validity
        valid = discriminator(word)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates words => determines validity
        combined = Model(inputs=noise, outputs=valid)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        print(generator.summary(), discriminator.summary(), combined.summary())
        return generator, discriminator, combined



if __name__ == "__main__":
    TestPolyGen = PolitiGen()
    TestPolyGen.build_combined()
