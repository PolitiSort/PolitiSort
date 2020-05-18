import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from PolitiSort.data.io import Tokenizer, DataHandler, GANHandler
    from PolitiSort.data import hydrate
    from PolitiSort.network import PolitiGen
import pickle
import sys
import argparse

parser = argparse.ArgumentParser("PolitiGen")
parser.add_argument("command", help="[scrape] dataset/[compile] corpus/[train] model", type=str)
parser.add_argument("-i", "--input", help="Input file path. Either the Corpus, Compiled Data, or Raw Acounts", type=str)
parser.add_argument("-o", "--output", help="Output file path. Either the Corpus, Compiled Data, or Network HDH5", type=str)
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
# net = PolitiGen(handler)
# net.train(iterations=10248, batch_size=128, reporting=8)

