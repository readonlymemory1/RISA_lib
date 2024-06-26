import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import sent_tokenize
# from transformers import pipeline
import re
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import itertools
from gensim.utils import tokenize
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric
import numpy as np

def custom_preprocess(doc):
    """
    텍스트를 소문자로 변환하고, 토큰화하며 2글자 이하의 단어도 포함합니다.
    """
    # 태그 제거, 구두점 제거, 다중 공백 제거, 숫자 제거
    doc = strip_tags(doc)
    doc = strip_punctuation(doc)
    doc = strip_multiple_whitespaces(doc)
    doc = strip_numeric(doc)

    # 소문자로 변환하고 토큰화
    tokens = list(tokenize(doc, lowercase=True))

    # 여기서는 2글자 이하의 단어도 포함
    return tokens


data_file = open("C:/Users/kjh05/OneDrive/문서/GitHub/RISA_lib/RISA_lib/data.data", "r")
data = data_file.read()
data_file.close()

a_h = pd.read_csv("C:/Users/kjh05/OneDrive/문서/GitHub/RISA_lib/RISA_lib/ai_human.csv", names=["ai", "human"])
# def ai_human(csv, in, out):
#     csv.loc[csv["ai"]==in, "human"] = out


def list_lower(i):
    for j in range(len(i)):
        i[j] = i[j].lower()
    return i

def list_re(i):
    for j in range(len(i)):
        i[j] = re.sub(r'[^a-zA-Z ]', "", i[j])
    return i

d = list(a_h["ai"])
data = data.lower()
data = data.replace("\n", " ")
data = sent_tokenize(data)+list_lower(d)
processed_sentences = [custom_preprocess(sentence) for sentence in data]
w2v_model = Word2Vec(sentences=processed_sentences, vector_size=100, window=5, min_count=1, workers=4)
# data = split_sentences(data)
# for i in range(len(data)):
#     data[i] = data[i].lower()
# print(data)
# 단어 사전 생성
# vocab = set(re.sub(r'[^a-zA-Z ]', "", ' '.join(data)).split())
sent = list(itertools.chain(*processed_sentences))
vocab_set = list(set(sent))

word_to_idx = {word: i+1 for i, word in enumerate(w2v_model.wv.index_to_key)}
idx_to_word = {i+1: word for i, word in enumerate(w2v_model.wv.index_to_key)}
word_to_idx["<eos>"] = 0
idx_to_word[0] = "<eos>"
vocab_size = len(word_to_idx)+1

input_sequences = []
for sentence in processed_sentences:
    for i in range(1, len(sentence)):
        n_gram_sequence = sentence[:i+1]
        input_sequences.append([word_to_idx[word] for word in n_gram_sequence])

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array([np.pad(seq, (max_sequence_len-len(seq), 0), 'constant') for seq in input_sequences])
X, y = input_sequences[:,:-1], input_sequences[:,-1]

x_train = torch.LongTensor(X)
y_train = torch.LongTensor(y)
# 문장을 숫자 시퀀스로 변환
def seq_to_indices(seq):
    return [word_to_idx[word] for word in custom_preprocess(seq)]

# 학습용 데이터셋 생성
sent_tokenize = sent_tokenize(" ".join(sent))
print(sent_tokenize)
sent_tokenize = list_re(sent_tokenize)
print(sent_tokenize)
# for seq in sent_tokenize:
sequences = [seq_to_indices(seq)+[0] for seq in sent_tokenize]
print(sequences)
# sequences = [seq_to_indices(seq) + [0] for seq in split_sentences(re.sub(r"[^a-zA-Z ]",""," ".join(data)))]  # 각 시퀀스 끝에 <eos> 추가
input_sequences = [torch.tensor(seq[:-1]) for seq in sequences]
target_sequences = [torch.tensor(seq[1:]) for seq in sequences]

# 하이퍼파라미터
embedding_dim = 100
hidden_dim = 32
num_layers = 1
num_epochs = 100
learning_rate = 0.01

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, weights):
        super(LSTMModel, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

weights = torch.FloatTensor(w2v_model.wv.vectors)
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, weights)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    # for input_seq, target_seq in zip(input_sequences, target_sequences):
    #     optimizer.zero_grad()
    #     hidden = (torch.zeros(num_layers, 1, hidden_dim),
    #               torch.zeros(num_layers, 1, hidden_dim))
    #
    #     output, _ = model(input_seq.unsqueeze(0), hidden)
    #     loss = criterion(output.view(-1, vocab_size), target_seq)
    #
    #     loss.backward()
    #     optimizer.step()
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 생성 함수 정의
def generate(model, start_seq, max_len=20):
    with torch.no_grad():
        words = start_seq.split()
        input_seq = torch.tensor([word_to_idx[word] for word in words], dtype=torch.long).unsqueeze(0)
        hidden = (torch.zeros(num_layers, 1, hidden_dim),
                  torch.zeros(num_layers, 1, hidden_dim))

        for _ in range(max_len):
            output, hidden = model(input_seq, hidden)
            last_word_logits = output[:, -1, :]
            _, top_word_idx = torch.max(last_word_logits, 1)
            if top_word_idx.item() == word_to_idx['<eos>']:  # <eos> 토큰을 만나면 종료
                break
            words.append(idx_to_word[top_word_idx.item()])
            input_seq = torch.cat((input_seq, top_word_idx.unsqueeze(0)), dim=1)

    return ' '.join(words)

# 모델을 사용하여 문장 생성
start_seq = "what"
generated_sentence = generate(model, start_seq)
print("Generated Sentence:", generated_sentence)
