from nltk.tokenize import WordPunctTokenizer
from Simple_Debug import Debug
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class train_tool:
    def __init__(self, input = ""):
        self.input_len = len(input)
        self.input = input
        self.word_list = []
        for word in WordPunctTokenizer().tokenize(input):
            self.word_list.append(word)
        self.word_set = set(self.word_list)

    def w2i(self):
        w2i = {w: i for i, w in enumerate(self.word_set)}
        w2i['<NUL>'] = 0
        return w2i
    def i2w(self):
        i2w = {i:w for i, w in enumerate(self.word_set)}
        i2w[0] = '<NUL>'
        return i2w
    # def talking_data_tool(self, file, person):
    #     f = open(file, "r")
    #     data = f.read()
    #     f.close()
    #     data = data.split
    #     person = []
    #     return data

    def vocab_list(self):
        input_list = str(self.input)
        vocab_list = list(set(input_list.split()))

        return vocab_list

    def build_data(self):
        encoded = [self.w2i()[token] for token in self.input.split()] # 각 문자를 정수로 변환.
        input_seq, label_seq = encoded[:-1], encoded[1:] # 입력 시퀀스와 레이블 시퀀스를 분리
        input_seq = torch.LongTensor(input_seq).unsqueeze(0) # 배치 차원 추가
        label_seq = torch.LongTensor(label_seq).unsqueeze(0) # 배치 차원 추가
        print(self.input)
        return input_seq, label_seq
    def encoder(self, x, start_input=False):
        if start_input:
            data_split = x.split()
            data = []
            for i in range(4):
                data.append(self.w2i()["<NUL>"])
            for i in range(len(data_split)):
                data.append(self.w2i()[data_split[i]])

        else:
            data_split = x.split()
            data = []
            for i in range(len(data_split)):
                data.append(self.w2i()[data_split[i]])
        return data



    def decoder(self, x):
        return self.i2w()[x]
    def using_for_decoder(self, x):
        data = []
        for i in range(len(x)):
            data.append(self.i2w()[x[i]])
        return data

#model_section
class test_model(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(test_model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.rnn_layer = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        output = self.embedding_layer(x)
        output, hidden  = self.rnn_layer(output)
        output = self.linear(output)
        return output.view(-1, output.size(2))
