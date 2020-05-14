import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from PolitiSort.data.io import Tokenizer, DataHandler, GANHandler
    from PolitiSort.network import PolitiGen
import pickle
import sys
import argparse

parser = argparse.ArgumentParser("PolitiSort")
parser.add_argument("command", help="[compile] corpus/[train] model/[scrape] dataset", type=int)
parser.add_argument("-i", "--input", help="Input file path. Either the Corpus, Compiled Data, or Raw Acounts", type=int)
parser.add_argument("-o", "--output", help="Output file path. Either the Corpus, Compiled Data, or Network HDH5", type=int)
parser.add_argument("--key", help="A JSON String acquired from Twitter shaped {'CONSUMER_KEY': '', 'CONSUMER_SECRET': '', 'ACCESS_KEY': '', 'ACCESS_SECRET': ''}")
args = parser.parse_args()

print(args)

# tokenizer = Tokenizer("./static/1billion_word_vectors")
# handler = GANHandler("./senators_twsc_long.csv", tokenizer)
# handler.compile()
#with open("DH.GANHandler", "wb") as df:
#    pickle.dump(handler, df)
# breakpoint()
# with open("DH.GANHandler", "rb") as df:
# r   handler = pickle.load(df)
# net = PolitiGen(handler)
# net.train(iterations=10248, batch_size=128, reporting=8)


