import csv

class Tokenization():
    def __init__(self):
        self.__vocab = {}
        self.__vocab_rev = {}
        self.__counter = 0

    def tokenize(self, string):
        arrayOfStrings = string.split()
        arrayOfNums = []
        for word in arrayOfStrings:
            token = self.__vocab.get(word)
            if not token:
                self.__vocab[word] = self.__counter
                self.__vocab_rev[self.__counter] = word
                token = self.__counter
                self.__counter += 1
                arrayOfNums.append(token)
        return arrayOfNums

    def getStrings(self, nums):
        arrayOfStrings = []
        for number in nums:
                string = self.__vocab_rev[number]
                arrayOfStrings.append(string)
        return arrayOfStrings


class DataHandler():
    def __init__(self, csvInput, tokenizer: Tokenization):
        self.tokenizer = tokenizer
        self.__csvInput = csvInput
        self.__encodedData = {}
        self.__isCompiled = False

    def __call__(self):
        return self.__encodedData

    def compile(self, retreiveFields):
        with open(self.__csvInput, 'r') as df:
            csvreader = csv.reader(df)
            for field in retreiveFields:
                arrayThing = []
                for row in csvreader:
                    arrayThing.append(self.tokenizer.tokenize(row[field]))
                    self.__encodedData.update({field:arrayThing})
        self.__isCompiled = True

