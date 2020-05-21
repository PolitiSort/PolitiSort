import tensorflow as tf
import math
import random
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from contextlib import redirect_stdout
from keras.layers import Input, Dense, Masking, LSTM, Lambda, Dropout, Conv1D, BatchNormalization, ZeroPadding1D, Flatten, UpSampling1D, Reshape, concatenate, LeakyReLU 
# Dependencies used in the version of Politisort that takes Bios into account
from keras.layers import UpSampling2D, Conv2D, LeakyReLU, Activation


import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from tqdm import tqdm
from .data.io import GANHandler


class PolitiGen(object):
    """
    Creates a GAN model to generate political tweets.
    Has methods to train the GAN.
    """

    def __init__(self, handler: GANHandler):
        """
        Creates a PolitiGen network
        :param handler: a compiled GANHandler. See PolitiSort.io.data
        """
        self.handler = handler
        self.gen, self.desc, self.comb = self.__compile()

    def __build_discriminator(self):
        """
        Put together a CNN that will return a single confidence output.

        :returns: the model object
        """
        # We seed a Sequential network
        model = Sequential()
        # Expand the sequential input to temporal convolution slices by expanding dims
        model.add(Lambda(lambda x: K.expand_dims(x, axis=-1))) # [1, 2, 3] => [[1], [2], [3]]
        # <Conv1D Net>
        model.add(Conv1D(64, 5, activation=LeakyReLU()))
        model.add(Conv1D(128, 3, activation=LeakyReLU()))
        # </Conv1D Net>
        # Flatten ConvNet into Sequential Object
        model.add(Flatten())
        # <DNN>
        model.add(Dense(64, activation=LeakyReLU()))
        model.add(Dense(16, activation=LeakyReLU()))
        model.add(Dense(8, activation=LeakyReLU()))
        model.add(Dense(1, activation='sigmoid'))
        # </DNN>
        return model

    def __build_generator(self):
        """
        Put together a model that takes in one-dimensional noise and outputs one-dimensional
        data representing words with each number representing an index in our corpus.

        :returns: the model object
        """

        # We seed a Sequential network
        model = Sequential()
        # Expand the sequential input to temporal convolution slices by expanding dims
        model.add(Lambda(lambda x: K.expand_dims(x, axis=-1))) # [1, 2, 3] => [[1], [2], [3]]
        # <Conv1D Net>
        model.add(Conv1D(128, 3))
        model.add(Conv1D(64, 5))
        # </Conv1D Net>
        # Upsample the data to force synthesis
        model.add(UpSampling1D(size=4))
        # <Conv1D Net>
        model.add(Conv1D(64, 7))
        model.add(Conv1D(32, 9))
        # </Conv1D Net>
        # Upsample the data to force synthesis
        model.add(UpSampling1D())
        # Flatten ConvNet into Sequential Object
        model.add(Flatten())
        # <DNN>
        model.add(Dense(568))
        model.add(Dense(256))
        model.add(Dense(100))
        # </DNN>
        return model

    def __compile(self):
        """
        Puts together a model that combines the discriminator and generator models.

        :returns: the generator, discriminator, and combined model objects
        """

        optimizer = Adam(5e-4)

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

        # Open the model summary file
        with open("model.summary", "w") as df:
            with redirect_stdout(df):
                combined.summary()
        return generator, discriminator, combined

    def __unison_shuffled_copies(self, a, b):
        """
        Unison shuffling two numpy arrays

        returns: a uniformly shuffled two numy arrays
        """

        # Check that the lengths are equal, otherwise this may not be very helpful
        assert len(a) == len(b)
        # Take a permutation nPn, which effectively returns a list of length len(a) in random order 
        p = np.random.permutation(len(a))
        # Re-Index inputs to that random list order
        return a[p], b[p]

    def save(self, directory):
        self.comb.save(directory)

    def synthesize(self, length=None):
        if not length:
            length = random.randint(5, 54)

        inp, _, _, _, _, _ = self.handler.step(2)
        generated_results = self.gen.predict(inp)
        generated_pairs = []
        for indx, e in enumerate(generated_results):
            generated_pairs.append(np.hstack([inp[indx], e])) # Stack original input and output together
        generated_pairs_translated = self.handler.translate(generated_pairs[0])
        sent = generated_pairs_translated[0] + " " + generated_pairs_translated[1]
        inp = generated_results
        for _ in range(length-1):
            generated_results = self.gen.predict(np.array(inp))
            inp = generated_results
            generated_pairs = []
            for indx, e in enumerate(generated_results):
                generated_pairs.append(np.hstack([inp[indx], e])) # Stack original input and output together
            generated_pairs_translated = self.handler.translate(generated_pairs[0])
            sent = sent + " " + generated_pairs_translated[1]

    def train(self, epochs=100, iterations=1024, batch_size=128, reporting=50):
        """
        Method trains the GAN.

        :param epochs: trains the GAN for a default of 100 epochs.
        :param iterations: trains the GAN with a default amount of 1024 iterations.
        :param batch_size: trains the GAN with a batch size of 128 by default.
        :param reporting: reports every 50 iterations by default.
        :return:
        """

        # For each epoch...
        for epoch in range(epochs):
            # Report the epoch head
            print("Epoch {}/{}".format(epoch+1, epochs))

            # Iterate though num_iter for sampling
            for i in tqdm(range(iterations)):
                # Sample a batch from the compiled DataHandler (see Politisort.data.io)
                inp, actual_pairs, zeros, ones, noise, full_ones = self.handler.step(batch_size)
                # Predict an output
                generated_results = self.gen.predict(inp)
                # Sew the data into pairs with the inputs given
                generated_pairs = []
                for indx, e in enumerate(generated_results):
                    generated_pairs.append(np.hstack([inp[indx], e])) # Stack original input and output together
                # Turn the aforemention generated_pairs to an numpy array
                generated_pairs = np.array(generated_pairs)
                # Shuffle input/outputs in unison
                desc_in, desc_out = self.__unison_shuffled_copies(np.vstack([actual_pairs, generated_pairs]), np.hstack([ones, zeros]))
                # Use discriminators to predict actual, generated pairs' identities for logging/validation
                d_res_real = self.desc.predict(actual_pairs)
                d_res_fake = self.desc.predict(generated_pairs)

                # Train the discriminator
                d_loss_fake = self.desc.train_on_batch(generated_pairs, zeros)
                d_loss_real = self.desc.train_on_batch(actual_pairs, ones)
                # Backprop the newly trained gradients using "fake ones"
                g_loss = self.comb.train_on_batch(noise, full_ones)

                # Translate the predicted vectors to human words for logging
                generated_pairs_translated = self.handler.translate(generated_pairs[0])

                # Log if need be
                if i%reporting == 0:
                    print("i={}, Disc loss (R): {}, Disc loss (F): {}, Gen loss: {}, Words: {}, Disc Out (Sample Fake): {}, Disc Out (Sample Real): {}".format(i, d_loss_real[0], d_loss_fake[0], g_loss, str(generated_pairs_translated), d_res_fake[0][0], d_res_real[0][0]))
                
# <Here lies what was PolitiNet, a Polticial Stance Sorting thing>
# class PolitiNet(object):
    # def __init__(self, handler:GANHandler, maxdesc=15, maxstatus=15, lr=5e-3):
        # self.__maxDesc = maxdesc
        # self.__maxStatus = maxstatus
        # self.modelGenerator = None
        # self.modelDiscriminator = None
        # self.model.compile(Adam(lr=lr), loss="binary_crossentropy", metrics=["acc"])
        # self.handler = handler

    # def __build(self):
        # # inp_bio = Input(shape=(self.__maxDesc,)) 

        # inp_status = Input(shape=(self.__maxStatus,)) 

        # inp_masked = Masking(mask_value=0.0)(inp_status)

        # # inp_exp_bio = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp_bio)

        # inp_exp_status = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp_masked)

        # net1 = LSTM(64, return_sequences=True, activation="tanh")(inp_exp_status)
        # net1 = LSTM(32, return_sequences=True, activation="tanh")(net1)
        # net1 = LSTM(16, activation="tanh")(net1)

        # net3 = Dense(128, activation="tanh")(net1)
        # net3 = Dense(32, activation="tanh")(net1)
        # net3 = Dense(32, activation="tanh")(net3)
        # net3 = Dense(2, activation="softmax")(net3)

        # model = Model(inputs=[inp_status], outputs=[net3]) 
        # return model

    # def fit(self, handler, epochs=10, batch_size=32, val_split=0.05):
        # input_data = handler()
        # self.model.fit(x=[pad_sequences(input_data["status"], maxlen=self.__maxStatus)], y=[input_data["isDem"]], epochs=epochs, batch_size=batch_size, validation_split=val_split, shuffle=True)
# </Here lies what was PolitiNet, a Polticial Stance Sorting thing>
