from . import network  # File that contains the network and all it's functions
from .data import hydrate  # File that contains the hydrator
import pickle # Pickle library


def scrape(input, output, key):
    '''
    Function scrape scrapes Twitter using the provided snowflakes into a provided file using a provided key

    :param input: The snowflakes of the accounts that are to be scraped for their tweets.
    :param output: The destination file that this data is to be scraped to.
    :param key: Key used to scrape Twitter.
    :return: Nothing
    '''

    hydrate.run(input, output, key)


def trainModel(epochs, iterations, batch_size, reporting_count, handler, modelSaveFile = "do not save"):
    '''
    Function trainModel trains and saves a model using the passed parameters

    :param epochs: Trains the model using the passed amount of epochs
    :param iterations: Trains the model using the passed amount of iterations every epoch
    :param batch_size: Trains the model using the passed batch size
    :param reporting_count: Reports the status of the model every given amount of iterations
    :param handler: Uses the given handler to tokenize input data
    :param modelSaveFile: Saves the model to this file if passed
    :return: returns the trained model if 3
    '''

    with open(handler, "rb") as df:
        GANhandler = pickle.load(df)
    net = network.PolitiGen(GANhandler)
    net.train(epochs, iterations, batch_size, reporting_count)
    if modelSaveFile != "do not save":
        net.save(modelSaveFile)
    return net


def compile(inputFile, outputFile):
    '''
    The compile function takes the passed CSV and creates a handler that will then be dumped into a passed pickle path

    :param inputFile: The path to the CSV file to be handled.
    :param outputFile: The path of the pickle that the handler will be dumped into.
    :return:
    '''

    pass


def generate(seed, input, sentence_count):
    pass
