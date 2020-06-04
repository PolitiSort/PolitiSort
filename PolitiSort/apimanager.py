from . import network  # File that contains the network and all it's functions
from .data import hydrate, io  # File that contains the hydrator
import pickle  # Pickle library


def scrape(users, output_file, key):
    '''
    Function scrape scrapes Twitter using the provided snowflakes into a provided file using a provided key

    :param users: An array of snowflakes of the accounts that are to be scraped for their tweets.
    :param output_file: The destination file that this data is to be scraped to.
    :param key: Key used to scrape Twitter.
    :return: Nothing
    '''
    hydrate.run(users, output_file, key)



def train_model(epochs, iterations, batch_size, reporting_count, handler, modelSaveFile = "do not save"):
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


def compile_handler(input_file, tokenizer, output_file="do not save"):
    '''
    The compile function takes the passed CSV and creates a handler that will then be dumped into a passed pickle path, or returned

    :param input_file: The path to the CSV file to be handled.
    :param output_file: The path of the pickle that the handler will be dumped into. If not provided it will not be pickled
    :return: Handler
    '''
    tokenizer = io.Tokenizer(tokenizer)
    handler = io.GANHandler(input_file, tokenizer)
    handler.compile()
    if output_file != "do not save":
        with open(output_file, "wb") as df:
            pickle.dump(handler, df)
    return handler


def generate_tweet(handler, model, sentence_count, output_file="none"):
    '''
    The generateTweet function creates tweets based upon how many sentences were desired

    :param handler: The path to the handler for tokenization (must be the same one used in the training of the model)
    :param model: A path to the model file
    :param sentence_count: How many sentences are to be generated
    :param output_file: An optional parameter being the path of an output file.
    :return: The generated tweet
    '''

    if type(handler) is str and type(model) is str:

        net = network.PolitiGen.load(model, handler)
        sents = ""

        for _ in range(sentence_count):
            sents = sents + net.synthesize() + " "
        if output_file != "none":
            with open(output_file, "w") as df:
                df.write(sents)
        return sents

    else:  # TODO allow for handler and model object inputs and change doc comment
        raise ValueError("Invalid path to model and or handler given")

def loadModel(handler, modelPath):
    '''
    Function to load a generator model of specified path
    :param handler: The handler to be used with the model
    :param modelPath: The path of the saved model
    :return: The model object
    '''
    return network.PolitiGen.load(modelPath, handler)

