from PolitiSort.data.io import Tokenizer, DataHandler, GANHandler
from PolitiSort.network import PolitiNet, PolitiGen
import pickle


# tokenizer = Tokenizer()
# handler = GANHandler("./senators_twsc_long.csv", tokenizer)
# handler.compile()
# with open("DH.GANHandler", "wb") as df:
    # pickle.dump(handler, df)
with open("DH.GANHandler", "rb") as df:
    handler = pickle.load(df)
# breakpoint()
net = PolitiGen()
net.train(handler, iterations=10248, batch_size=128, reporting=256)

