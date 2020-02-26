%tensorflow_version 2.x
%matplotlib inline
from gensim.models import word2vec
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

(x_train, y_train), (x_test, y_test) = jacksdata.load_data()


x_train[0]


political_offset = 3
political_map = dict((index + jacksdata_offset, word) for (word, index) in jacksdata.get_word_index().items())
political_map[0] = 'PADDING'
political_map[1] = 'START'
political_map[2] = 'UNKNOWN'


' '.join([political_map[word_index] for word_index in x_train[0]])



train_sentences = [['PADDING'] + [political_map[word_index] for word_index in review] for review in x_train]
test_sentences = [['PADDING'] + [political_map[word_index] for word_index in review] for review in x_test]


political_wv_model = word2vec.Word2Vec(train_sentences + test_sentences + ['UNKNOWN'], min_count=1, size=100)



political_wordvec = political_wv_model.wv
del political_wv_model



#processing
lengths = [len(tweet) for tweet in x_train]
lengths2 = [len(tweet) for tweet in x_test]
print('Longest tweet: {} Shortest tweet: {}'.format(max(lengths), min(lengths)))
print('Longest tweet: {} Shortest tweet: {}'.format(max(lengths2), min(lengths2)))



tweetLengths = [len(article) for article in np.hstack([x_test, x_train])]
plt.hist(tweetLengths, 50)



cutoff = 140
print('{} tweets out of {} are over {}.'.format(
    sum([1 for length in lengths if length > cutoff]), 
    len(lengths), 
    cutoff))
print('{} tweets out of {} are over {}.'.format(
    sum([1 for length in lengths2 if length > cutoff]), 
    len(lengths2), 
    cutoff))



from keras.preprocessing import sequence
x_train_padded = sequence.pad_sequences(x_train, maxlen=cutoff)
x_test_padded = sequence.pad_sequences(x_test, maxlen=cutoff)



#classification
from keras.layers import Conv2D, MaxPooling2D
model = Model(inputs=[]
embedding = model.add(Embedding(input_dim=len(political_map), output_dim=200, input_length=cutoff))

2ndboi = keras.layers.LSTM(units, activation='relu', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False) (embedding)
2ndboi = keras.layers.LSTM(units, activation='relu', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False) (2ndboi)








