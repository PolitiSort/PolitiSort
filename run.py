from PolitiSort.data.io import Tokenizer, DataHandler
from PolitiSort.network import PolitiNet


tokenizer = Tokenizer()
handler = DataHandler("./senators_twsc_long.csv", tokenizer)
handler.compile(["handle", "name", "description", "status", "isDem"])
net = PolitiNet(32,32)
net.fit(handler, epochs=1000, batch_size=128, val_split=0.00)


