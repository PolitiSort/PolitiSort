import csv
import random
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

class Tokenizer(object):
    def __init__(self, embedding_vector_file=""):
        self.__vocab = {0:"{{URL}}"}
        self.__vocab_rev = {"{{URL}}":0}
        self.__counter = 1
        self.__embedding = Word2Vec.load(embedding_vector_file)

    @property
    def _counter(self):
        return self.__counter

    def get_word(self, vector:np.ndarray):
        word = self.__embedding.similar_by_vector((vector*10)-10)[0][0]
        return word

    def conform(self, vector:np.ndarray):
        word = self.__embedding.similar_by_vector((vector*10)-10)[0][0]
        return self.__embedding[word]

    def tokenize(self, string, by_char=False):
        arrayOfStrings = string.split() if not by_char else list(string)
        arrayOfNums = []
        for word in arrayOfStrings:
            word = word.strip().strip(".").strip("!").strip("?").strip("/").strip("-")
            if "https" in word or "…" in word:
                continue
            try:
                token = ((self.__embedding[word.lower()])+10)/10
            except KeyError:
                continue
            arrayOfNums.append(token)
        return np.array(arrayOfNums)

    def getString(self, nums):
        arrayOfStrings = []
        for number in nums:
                string = self.get_word(number)
                arrayOfStrings.append(string)
        return arrayOfStrings


class GANHandler(object):
    def __init__(self, csvInput, tokenizer:Tokenizer, batch_size:int=32):
        self.tokenizer = tokenizer
        self.__csvInput = csvInput
        self.__encodedData = defaultdict(list)
        self.__isCompiled = False

    def noise(self, batch:int):
        samples = self.__encodedData["statuswords"]
        indexes = np.random.randint(0,samples.shape[0],(batch,))
        return samples[indexes]

    @staticmethod
    def __noisy_labels(batch_size, wrong_prob=0.1, correct=0, incorrect=1):
        res = []
        for _ in range(batch_size):
            seed = random.uniform(0,1)
            if wrong_prob >= seed:
                res.append(incorrect)
            else:
                res.append(correct)
        return np.array(res)

    def translate(self, gen_pairs:list):
        gen_pairs = np.split(gen_pairs, 2)
        return [self.tokenizer.get_word(e) for e in gen_pairs]

    def conform(self, gen_results:list):
        return [self.tokenizer.conform(e) for e in gen_results]

    def step(self, batch_size):
        assert self.__isCompiled, "Uncompiled Handler! Call GANHandler().compile()"
        halfbatch = int(batch_size/2)
        assert halfbatch == batch_size/2, "Batch size must be divisible by 2!!"
        prev = self.noise(halfbatch)
        new_indxs = np.random.randint(1, len(self.__encodedData["bigrams"])-1, halfbatch)
        new = self.__encodedData["bigrams"][new_indxs]
        zeros = self.__noisy_labels(halfbatch)
        ones = self.__noisy_labels(halfbatch, correct=1, incorrect=0)
        full_ones = np.ones(batch_size)
        return prev, new, zeros, ones, self.noise(batch_size), full_ones
        
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
                check = np.hstack([self.__encodedData["status"][indx][e], self.__encodedData["status"][indx][e+1]])
                self.__encodedData["bigrams"].append(check)
            
        self.__encodedData["bigrams"] = np.array(self.__encodedData["bigrams"])
        self.__encodedData["statuswords"] = np.array([item for sublist in self.__encodedData["status"] for item in sublist])
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

