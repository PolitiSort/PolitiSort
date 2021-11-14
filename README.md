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

The next step is to install all the dependencies of this project. You can do this with Conda (we recommend using Conda anyways) by doing:

```bash
conda create --name PolitiSort --file packages.txt
```

There is no current easy way to install all the dependencies with pip, so unfortunately pip users will have to manually install each dependency. Alternatively, you could use Conda! 

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

<<<<<<< HEAD
*Note: inputs to the Politisort model will be tokenized and outputs of the GAN model will also be tokenized. If saving those models, it may be wise to save the token library as well in the io.py file under Politisort/data.*

## API details
As already noted, there is currently no API support for the Politisort portion of this project, but for the Politigen portion, we have developed a fairly comprehensive API.
As an example to show how this should be used, let's walk through training and saving and using a model. Note that more details can be found in the documentation comments of each function. To start, let's scrape some tweets. Assuming we already have an array of user account snowflakes, we would do this by doing:
```python
from PolitiSort.apimanager import scrape
user_snowflakes = [] # For the purposes of this demonstration we are going to pretend that has an array of integer user snowflakes
API_key = "seenoevil" # This would be a valid api key
scrape(user_snowflakes, "tweets.csv", API_key)
```
Now that we have a CSV containing all of the tweets, let's compile a GANHandler for our network. At this point we will tokenize based upon the 1billion word vectors dataset, but you could theoretically use any pickled word-embedding trained using the Gensim Wordnet package. These can be found online, or potentially on your computer if you've already downloaded one.
```python
from PolitiSort.apimanager import compile_handler
Tokenizer = "./static/1billion_word_vectors"
handler = compile_handler("tweets.csv", Tokenizer, "GANHandler") # By excluding the last argument, this function does not save the handler to a file and only returns it
```
Now we have all the necessary components to train the model, let's run the train function!
```python
from PolitiSort.apimanager import train_model
epochs = 400
iterations = 1000
batch_size = 120
reporting_count = 10
handler = "./GANHandler" # This can be either a GANHandler object, or a file path
output_file = "./politigenv1trained" # This is an optional argument and can be excluded if you wish for the model to be only returned
train_model(epochs, iterations, batch_size, reporting_count, handler, output_file)
```
Now that we've trained the model and saved it to a file for future use, let's use the API to generate some tweet sentences, but to do that we first need to load the model.
```python
from PolitiSort.apimanager import loadModel
handler = "./GANHandler" # We once again will use our handler
model_path = "./politigenv1trained"
model = loadModel(handler, model_path)
```
Now that we have loaded the model, it is finally time to generate some sentences!
```python
from PolitiSort.apimanager import generate_tweet
handler = "./GANHandler" 
sentence_count = 5
output_file = "generated_tweets.txt" # This is once again an optional parameter
generate_tweet(handler, model, sentence_count, output_file)
```
Now you should know how to use our API to train and use a model. Thank you for using Politigen!

--with :heart: and :green_salad: from @Jemoka ond @zbuster05

