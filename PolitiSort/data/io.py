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

if __name__ == "__main__":
	testTokenConverter = Tokenization()
	tokens = testTokenConverter.tokenize("hi my name is bob and I like cheese")
	print(tokens)
	print(testTokenConverter.getStrings(tokens))
