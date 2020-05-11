# PolitiSort
Sort and generate your stance, automagically with Politisort. 

## Overview

Politisort is a project containing two seperate efforts. The Politisort part of this project aims to detect the political stance of a tweet. The Politigen part of this project aims to generate politically biased tweets. Both of these parts have been trained off of a dataset created by scraping the twitter account of each current senator.




## Setup and Installation
The first step is to clone the repository using  the following command:

```git clone <insert url because i'm too lazy right now> ```

Next, change directorys into the repository using:

```cd Politisort```

The next step is to install all the dependencies of this project. You can do this with Conda by doing:

``` conda create --name Politisort --file packages.txt```

There is no current easy way to install all the dependencies with pip, so i'm afraid that pip users will have to manually install each dependency. Alternatively, you could use Conda!
At this point you should be ready to run the project. All that needs to be done in that case is the running run.py using:

```python3 run.py```

This should automatically begin training the GAN model in this project. If you encounter errors, solve them.

## Politigen vs Politisort

This repository has two different projects: Politisort and Politigen. At the moment, the run.py file will compile and train the Politigen model in the master branch. If you wish to train and run the Politisort model, checkout to the Politisort branch. In the future, these branches are likely to be merged and run.py will likely have system arguments. Running run.py in any branch using the following command will train the model:

```python3 run.py```

To store the trained model, refer to the Keras documentation.