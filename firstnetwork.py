%tensorflow_version 2.x
%matplotlib inline
from gensim.models import word2vec


(x_train, y_train), (x_test, y_test) = jacksdata.load_data()


x_train[0]


political_offset = 3
political_map = dict((index + jacksdata_offset, word) for (word, index) in jacksdata.get_word_index().items())
political_map[0] = 'PADDING'
political_map[1] = 'START'
political_map[2] = 'UNKNOWN'



