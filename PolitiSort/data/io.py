import csv
from collections import defaultdict

class Tokenizer(object):
    def __init__(self):
        self.__vocab = {}
        self.__vocab_rev = {}
        self.__counter = 0

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

