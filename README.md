# PolitiSort
Sort and generate your stance, automagically with PolitiSort. 

## Overview
PolitiSort is a project containing two seperate efforts. The PolitiSort part of this project aims to detect the political stance of a tweet. The PolitiGen part of this project aims to generate politically biased tweets. Both of these parts have been trained off of a dataset created by scraping the twitter account of each current senator.


## Setup and Installation

The first step is to clone the repository using  the following command:

```bash
git clone https://github.com/PolitiSort/PolitiSort.git
 ```

Change into that folder like so:

```bash
cd PolitiSort
```

The next step is to install all the dependencies of this project. You can do this with Conda (recommended, we use Conda anyways) by doing:

```bash
conda create --name PolitiSort --file packages.txt
```

There is no current easy way to install all the dependencies with pip, so I'm afraid that pip users will have to manually install each dependency. Alternatively, you could use Conda! 

**A Note About Numpy**: some of us have experienced issues with the model actually running when on the Numpy version in the dependencies list. The error message usually goes something about the fact that *Numpy can't load a pickle when `allow_pickle=False`*. Unfortunately, gensim, the topic modeling package we use, can't seem to play well with Numpy on some versions. The way we fixed it is to simply navigate to the offending file (inside the gensim package!), and change that line to `allow_pickle=True.` This should only occur when training a new GANHandler.

**A Note About GANHandler**: this project requires a compiled data format, GANHandler, to train the model. The file is a pickle containing an instance of the GANHandler class, which includes utility functions to load the data, generate bigrams, encode/decode the data, etc. This could be created from run.py.

At this point you should be ready to run the project. All that needs to be done in that case is the running run.py. From that script, you should be able to...

1. Create new datasets from a list of Twitter users
2. Compile new GANNHandlers from dataset CSVs
3. Train a model!
4. Make sentences with the model

The CLI syntax on how to do all of that could be found using the command:

```
python3 run.py -h
```


If you encounter errors, solve them. We believe in you!

## PolitiGen vs PolitiSort

This repository has two different projects: PolitiSort and PolitiGen. The former is a LSTM-based network to do political stance sorting, the latter a GAN to generate tweets using a Markov chain. At the moment, the run.py file will work with the PolitiGen model in the master branch. If you wish to train and run the PolitiSort model, checkout to the PolitiSort branch. Unfortunately we won't be building a CLI for PolitiSort at the moment, but you could still use the reasonably commented API in PolitiSort.network.PolitiNet.

To store the trained model, use the Keras save trainable weights method like this:

```python
myTrainedModel.save_weights('my_model_weights.h5')
```

Refer to the [Keras documentation](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models) for more details. For storing the PolitiSort model, you would likely want to store the model under PolitiNet and simply use the model predict method as shown below:

```python
PolitiSortNetwork = PolitNet()
model = PolitiSortNetwork.model
model.load_weights('my_model.h5')
model.predict(someValidTokenizedTweet)
```

*Note: inputs to the PolitiSort model will be tokenized and outputs of the GAN model will also be tokenized. If saving those models, it may be wise to save the token library as well in the io.py file under PolitiSort/data.*


--with :heart: and :green_salad: from @Jemoka ond @zbuster05
