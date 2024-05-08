from nltk.tokenize import WordPunctTokenizer

class train_tool:
    def __init__(self, input):
        self.input_len = len(input)
        self.input = input
        self.word_list = []
        for word in WordPunctTokenizer().tokenize(input):
            self.word_list.append(word)

    def w2i(self):
        w2i = {w: i for i, w in enumerate(self.word_list)}
        return w2i
    def i2w(self):
        w2i = {i:w for w, i in enumerate(self.word_list)}
tt = train_tool("don't go gentle into that good night")
print(tt.w2i())
        