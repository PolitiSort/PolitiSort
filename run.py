from PolitiSort.data.io import Tokenizer, DataHandler, GANHandler
from PolitiSort.network import PolitiNet


tokenizer = Tokenizer()
handler = GANHandler("./senators_twsc_long.csv", tokenizer)
handler.compile()
handler.step(16)
breakpoint()
net = PolitiNet(46,46)
net.fit(handler, epochs=1000, batch_size=128, val_split=0.00)


