import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from PolitiSort.data.io import Tokenizer, DataHandler, GANHandler
    from PolitiSort.data import hydrate
    from PolitiSort.network import PolitiGen

import pickle
# TODO ask jack why this is imported
import sys
import argparse

parser = argparse.ArgumentParser("PolitiGen")
parser.add_argument("command", help="[scrape] dataset/[compile] corpus/[train] model/[make] sentences", type=str)
parser.add_argument("-i", "--input", help="Input file path. Either the Corpus, Compiled Data, or Raw Acounts", type=str)
parser.add_argument("-o", "--output", help="Output file path. Either the Corpus, Compiled Data, or Network HDH5", type=str)
parser.add_argument("--trainargs", help="String shaped \"epochs iterations batch_size reporting_count\"", type=str)
parser.add_argument("-s", "--seed", help="Seed model for synthesis.", type=str)
parser.add_argument("--sentcount", help="The number of sentences to make.", type=int)
parser.add_argument("--key", help="A JSON String acquired from Twitter shaped {'CONSUMER_KEY': '', 'CONSUMER_SECRET': '', 'ACCESS_KEY': '', 'ACCESS_SECRET': ''}", type=str)
args = parser.parse_args()

command = args.command
if command == "scrape":
    hydrate.run(args.input, args.output, args.key)
elif command == "compile":
    tokenizer = Tokenizer("./static/1billion_word_vectors")
    handler = GANHandler(args.input, tokenizer)
    handler.compile()
    with open(args.output, "wb") as df:
        pickle.dump(handler, df)
elif command == "train":
    with open(args.input, "rb") as df:
        handler = pickle.load(df)
    net = PolitiGen(handler)
    if args.trainargs:
        ta = args.trainargs.split(" ")
        try:
            epochs = int(ta[0])
            iterations = int(ta[1])
            batch_size = int(ta[2])
            reporting_count = int(ta[3])
        except IndexError:
            raise ValueError("Malformed trainargs. Please shape it \"iterations batch_size reporting_count\".")
    else:
        epochs = int(input("Epochs: "))
        iterations = int(input("Iterations: "))
        batch_size = int(input("Batch Size: "))
        reporting_count = int(input("Reporting/Iter: "))
    net.train(epochs=epochs, iterations=iterations, batch_size=batch_size, reporting=reporting_count)
    net.save(args.output)
elif command == "make":
    net = PolitiGen.load(args.seed, args.input)
    sents = ""
    for _ in range(args.sentcount):
        sents = sents + net.synthesize() + " "
    with open(args.output, "w") as df:
        df.write(sents)

