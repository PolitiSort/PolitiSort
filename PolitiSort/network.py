import math
import random
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from contextlib import redirect_stdout
from keras.layers import Input, Dense, Masking, LSTM, Lambda, Dropout, Conv1D, BatchNormalization, ZeroPadding1D, Flatten, UpSampling1D, Reshape, concatenate
# Dependencies used in the version of Politisort that takes Bios into account
# from keras.layers import concatenate, LeakyReLU
from keras.layers import UpSampling2D, Conv2D, LeakyReLU, Activation


import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from tqdm import tqdm
from .data.io import GANHandler


class PolitiNet(object):
    def __init__(self, handler:GANHandler, maxdesc=15, maxstatus=15, lr=5e-3):
        self.__maxDesc = maxdesc
        self.__maxStatus = maxstatus
        self.modelGenerator = None
        self.modelDiscriminator = None
        self.model.compile(Adam(lr=lr), loss="binary_crossentropy", metrics=["acc"])
        self.handler = handler

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

class PolitiGen(object):
    def __init__(self, handler:GANHandler):
        self.handler = handler
        self.gen, self.desc, self.comb = self.__compile()

    def __build_discriminator(self):
        """
        Put together a CNN that will return a single confidence output.

        returns: the model object
        """
        model = Sequential()
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def __build_generator(self):
        """
        Put together a model that takes in one-dimensional noise and outputs one-dimensional
        data representing words with each number representing an index in our corpus.

        returns: the model object
        """
        noise_shape = (100,)

        model = Sequential()
        model.add(Lambda(lambda x: K.expand_dims(x, axis=-1)))
        model.add(Conv1D(64, 8, activation="relu"))
        model.add(Conv1D(32, 16, activation="relu"))
        model.add(UpSampling1D())
        model.add(Dropout(0.25))
        model.add(Conv1D(64, 8, activation="relu"))
        model.add(Conv1D(32, 16, activation="relu"))
        model.add(UpSampling1D())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        return model

    def __compile(self):
        """
        Puts together a model that combines the discriminator and generator models.

        returns: the generator, discriminator, and combined model objects
        """

        optimizer = Adam(6e-4)

        # Build and compile the discriminator
        discriminator = self.__build_discriminator()
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])

        # Build and compile the generator
        generator = self.__build_generator()

        # The generator takes noise as input and generates words
        noise = Input(shape=(100,))
        word = generator(noise)

        # For the combined model we will only train the generator
        discriminator.trainable = False

        # The discriminator takes generated words as input and determines validity
        validity = discriminator(concatenate([noise, word]))

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates words => determines validity
        combined = Model(inputs=noise, outputs=validity)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        # print(generator.summary(), discriminator.summary(), combined.summary())
        with open("model.summary", "w") as df:
            with redirect_stdout(df):
                combined.summary()
        return generator, discriminator, combined

    def __unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def train(self, epochs=100, iterations=1024, batch_size=128, reporting=50):

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            for i in tqdm(range(iterations)):
                inp, actual_pairs, zeros, ones, noise, full_ones = self.handler.step(batch_size)
                generated_results = self.gen.predict(inp)
                generated_pairs = []
                for indx, e in enumerate(generated_results):
                    generated_pairs.append(np.hstack([inp[indx], e]))
                generated_pairs = np.array(generated_pairs)
                desc_in, desc_out = self.__unison_shuffled_copies(np.vstack([actual_pairs, generated_pairs]), np.hstack([ones, zeros]))
                d_res_real = self.desc.predict(actual_pairs)
                d_res_fake = self.desc.predict(generated_pairs)
                d_loss_real = self.desc.train_on_batch(actual_pairs, ones)
                d_loss_fake = self.desc.train_on_batch(generated_pairs, zeros)
                g_loss = self.comb.train_on_batch(noise, full_ones)

                generated_pairs_translated = self.handler.translate(generated_pairs[0])

                if i%reporting == 0:
                    if not generated_pairs_translated[1]:
                        breakpoint()
                    print("i={}, Disc loss (R): {}, Disc loss (F): {}, Gen loss: {}, Words: {}, Disc Out (Sample Fake): {}, Disc Out (Sample Real): {}".format(i, d_loss_real[0], d_loss_fake[0], g_loss, str(generated_pairs_translated), d_res_fake[0][0], d_res_real[0][0]))
                    # breakpoint()
                

if __name__ == "__main__":
    TestPolyGen = PolitiGen()
