# PolitiSort
Sort and generate your stance, automagically with Politisort. 

## Overview

Politisort is a project containing two seperate efforts. The Politisort part of this project aims to detect the political stance of a tweet. The Politigen part of this project aims to generate politically biased tweets. Both of these parts have been trained off of a dataset created by scraping the twitter account of each current senator.




## Setup and Installation
The first step is to clone the repository using  the following command:

```
git clone https://github.com/PolitiSort/PolitiSort.git
 ```

Next, change directorys into the repository using:

```
cd Politisort
```

The next step is to install all the dependencies of this project. You can do this with Conda by doing:

``` 
conda create --name Politisort --file packages.txt
```

There is no current easy way to install all the dependencies with pip, so i'm afraid that pip users will have to manually install each dependency. Alternatively, you could use Conda!
At this point you should be ready to run the project. All that needs to be done in that case is the running run.py using:

```
python3 run.py
```

This should automatically begin training the GAN model in this project. If you encounter errors, solve them.

## Politigen vs Politisort

This repository has two different projects: Politisort and Politigen. At the moment, the run.py file will compile and train the Politigen model in the master branch. If you wish to train and run the Politisort model, checkout to the Politisort branch. In the future, these branches are likely to be merged and run.py will likely have system arguments. Running run.py in any branch using the following command will train the model:

```
python3 run.py
```

To store the trained model, use the Keras save trainable weights method like this:

```python
myTrainedModel.save_weights('my_model_weights.h5')
```

Refer to the [Keras documentation](https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models) for more details. In the case of the GAN, you would likely wish to purely store the generator model under the politigen class. For storing the Politisort model, you would likely want to store the model under Politinet and simply use the model predict method as shown below:

```python
PolitisortNetwork = Politnet()
model = PolitisortNetwork.model
model.load_weights('my_model.h5')
model.predict(someValidTokenizedTweet)
```

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
