from PolitiSort.data.io import Tokenizer, DataHandler, GANHandler
from PolitiSort.network import PolitiNet, PolitiGen
import pickle


tokenizer = Tokenizer("./1billion_word_vectors")
handler = GANHandler("./senators_twsc_long.csv", tokenizer)
handler.compile()
#with open("DH.GANHandler", "wb") as df:
#    pickle.dump(handler, df)
# breakpoint()
# with open("DH.GANHandler", "rb") as df:
# r   handler = pickle.load(df)
net = PolitiGen(handler)
net.train(iterations=1024, batch_size=256, reporting=16)


