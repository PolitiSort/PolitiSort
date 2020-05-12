# PolitiSort
Sort and generate your stance, automagically with Politisort. 

## Overview

Politisort is a project containing two seperate efforts. The Politisort part of this project aims to detect the political stance of a tweet. The Politigen part of this project aims to generate politically biased tweets. Both of these parts have been trained off of a dataset created by scraping the twitter account of each current senator.




## Setup and Installation
The first step is to clone the repository using  the following command:

```
git clone <insert url because i'm too lazy right now>
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