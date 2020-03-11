import csv
import numpy as np
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

class Tokenizer(object):
    def __init__(self):
        self.__vocab = {}
        self.__vocab_rev = {}
        self.__counter = 1

    @property
    def _counter(self):
        return self.__counter

    def _get_word(self, index:int):
        return self.__vocab_rev.get(int(index))

    def tokenize(self, string, by_char=False):
        arrayOfStrings = string.split() if not by_char else list(string)
        arrayOfNums = []
        for word in arrayOfStrings:
            token = self.__vocab.get(word.lower())
            if not token:
                self.__vocab[word.lower()] = self.__counter
                self.__vocab_rev[self.__counter] = word.lower()
                token = self.__counter
                self.__counter += 1
                arrayOfNums.append(token)
            else:
                arrayOfNums.append(token)
        return arrayOfNums

    def getString(self, nums):
        arrayOfStrings = []
        for number in nums:
                string = self.__vocab_rev[number]
                arrayOfStrings.append(string)
        return arrayOfStrings


class GANHandler(object):
    def __init__(self, csvInput, tokenizer:Tokenizer, batch_size:int=32):
        self.tokenizer = tokenizer
        self.__csvInput = csvInput
        self.__encodedData = defaultdict(list)
        self.__isCompiled = False

    @staticmethod
    def noise(batch:int):
        return np.random.randint(0, 20000, (batch,))

    def translate(self, gen_pairs:list):
        return [self.tokenizer._get_word(e*1e5) for e in gen_pairs]

    def step(self, batch_size, prev=None):
        assert self.__isCompiled, "Uncompiled Handler! Call GANHandler().compile()"
        halfbatch = int(batch_size/2)
        assert halfbatch == batch_size/2, "Batch size must be divisible by 2!!"
        if not prev:
            prev = self.noise(halfbatch)
        new_indxs = np.random.randint(1, len(self.__encodedData["bigrams"])-1, halfbatch)
        new = self.__encodedData["bigrams"][new_indxs]
        zeros = np.zeros(halfbatch)
        ones = np.ones(halfbatch)
        full_ones = np.ones(batch_size)
        return prev*1e-5, new*1e-5, zeros, ones, self.noise(batch_size)*1e-5, full_ones
        
    def compile(self, retreiveFields=["status"]):
        with open(self.__csvInput, 'r') as df:
            csvreader = csv.DictReader(df)
            for row in csvreader:
                for field in retreiveFields:
                    if field == "isDem":
                        self.__encodedData[field].append([0,1] if row['isDem']=='1' else [1,0])
                    elif field == "status" or field == "description":
                        self.__encodedData[field].append(self.tokenizer.tokenize(row[field]))
                    else:
                        self.__encodedData[field].append(self.tokenizer.tokenize(row[field], by_char=True))
        for indx, i in tqdm(enumerate(self.__encodedData["status"]), total=len(self.__encodedData["status"])):
            for e in range(len(i)-2):
                check = [self.__encodedData["status"][indx][e], self.__encodedData["status"][indx][e+1]]
                if check not in self.__encodedData["bigrams"]:
                    self.__encodedData["bigrams"].append(check)
        self.__encodedData["bigrams"] = np.array(self.__encodedData["bigrams"])
        # self.__encodedData["status"] = pad_sequences(self.__encodedData["status"], maxlen)
        # self.__encodedData["description"] = pad_sequences(self.__encodedData["description"], maxlen)

        self.__isCompiled = True


class DataHandler(object):
    def __init__(self, csvInput, tokenizer:Tokenizer):
        self.tokenizer = tokenizer
        self.__csvInput = csvInput
        self.__encodedData = defaultdict(list)
        self.__isCompiled = False

    def __call__(self):
        assert self.__isCompiled, "Uncompiled Handler! Call DataHandler().compile()"
        return dict(self.__encodedData)

    def compile(self, retreiveFields):
        with open(self.__csvInput, 'r') as df:
            csvreader = csv.DictReader(df)
            for row in csvreader:
                for field in retreiveFields:
                    if field == "isDem":
                        self.__encodedData[field].append([0,1] if row['isDem']=='1' else [1,0])
                    elif field == "status" or field == "description":
                        self.__encodedData[field].append(self.tokenizer.tokenize(row[field]))
                    else:
                        self.__encodedData[field].append(self.tokenizer.tokenize(row[field], by_char=True))
        self.__isCompiled = True

