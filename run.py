from PolitiSort.data.io import Tokenizer, DataHandler, GANHandler
from PolitiSort.network import PolitiNet
import pickle


# tokenizer = Tokenizer()
# handler = GANHandler("./senators_twsc_long.csv", tokenizer)
# handler.compile()
# with open("DH.GANHandler", "wb") as df:
    # pickle.dump(handler, df)
with open("DH.GANHandler", "rb") as df:
    handler = pickle.load(df)
print(handler.step(4))
breakpoint()
net = PolitiNet(46,46)
net.fit(handler, epochs=1000, batch_size=128, val_split=0.00)


