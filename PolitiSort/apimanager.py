from . import network  # File that contains the network and all it's functions
from .data import hydrate, io  # File that contains the hydrator
import pickle  # Pickle library


def scrape(input, output, key):
    '''
    Function scrape scrapes Twitter using the provided snowflakes into a provided file using a provided key

    :param input: The snowflakes of the accounts that are to be scraped for their tweets.
    :param output: The destination file that this data is to be scraped to.
    :param key: Key used to scrape Twitter.
    :return: True or False depending on if it works or not
    '''
    try:
        hydrate.run(input, output, key)
        return True
    except:
        return False


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
    if type(handler) is io.GANHandler:
        with open(handler, "rb") as df:
            handle = pickle.load(df)
    elif type(handler) is str:
        handle = handler
    else:
        raise ValueError("trainModel was passed neither a type GANHandler, nor filepath to Handler")

    net = network.PolitiGen(handle)
    net.train(epochs, iterations, batch_size, reporting_count)

    if modelSaveFile != "do not save":
        net.save(modelSaveFile)
    return net


def compile(inputFile, outputFile = "do not save"):
    '''
    The compile function takes the passed CSV and creates a handler that will then be dumped into a passed pickle path, or returned

    :param inputFile: The path to the CSV file to be handled.
    :param outputFile: The path of the pickle that the handler will be dumped into. If not provided it will not be pickled
    :return: Handler
    '''
    tokenizer = io.Tokenizer("./static/1billion_word_vectors")
    handler = io.GANHandler(inputFile, tokenizer)
    handler.compile()
    if outputFile != "do not save":
        with open(outputFile, "wb") as df:
            pickle.dump(handler, df)
    return handler


def generate(seed, input, sentence_count):
    pass
