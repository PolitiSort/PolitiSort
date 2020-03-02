from PolitiSort.data.io import Tokenizer, DataHandler
from PolitiSort.network import PolitiNet


tokenizer = Tokenizer()
handler = DataHandler("./senators_twsc.csv", tokenizer)
handler.compile(["handle", "name", "description", "status", "isDem"])
net = PolitiNet()
net.fit(handler, epochs=100, batch_size=16)


