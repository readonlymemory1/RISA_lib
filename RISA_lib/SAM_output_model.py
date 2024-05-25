import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import sent_tokenize
from transformers import pipeline
# from kss import split_sentences
import re

# 학습용 데이터 생성
data = ''''''

data = data.replace("\n", " ")
data = sent_tokenize(data)
# data = split_sentences(data)
# for i in range(len(data)):
#     data[i] = data[i].lower()
print(data)
# 단어 사전 생성
vocab = set(re.sub(r'[^a-zA-Z ]', "", ' '.join(data)).split())
word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
idx_to_word = {i+1: word for i, word in enumerate(vocab)}
word_to_idx["<eos>"] = 0
idx_to_word[0] = "<eos>"
vocab_size = len(word_to_idx)

# 문장을 숫자 시퀀스로 변환
def seq_to_indices(seq):
    return [word_to_idx[word] for word in seq.split()]

# 학습용 데이터셋 생성
sequences = [seq_to_indices(seq) + [0] for seq in sent_tokenize(re.sub(r"[^a-zA-Z ]",""," ".join(data)))]
# sequences = [seq_to_indices(seq) + [0] for seq in split_sentences(re.sub(r"[^a-zA-Z ]",""," ".join(data)))]  # 각 시퀀스 끝에 <eos> 추가
input_sequences = [torch.tensor(seq[:-1]) for seq in sequences]
target_sequences = [torch.tensor(seq[1:]) for seq in sequences]

# 하이퍼파라미터
embedding_dim = 10
hidden_dim = 32
num_layers = 1
num_epochs = 100
learning_rate = 0.01

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    for input_seq, target_seq in zip(input_sequences, target_sequences):
        optimizer.zero_grad()
        hidden = (torch.zeros(num_layers, 1, hidden_dim),
                  torch.zeros(num_layers, 1, hidden_dim))

        output, _ = model(input_seq.unsqueeze(0), hidden)
        loss = criterion(output.view(-1, vocab_size), target_seq)

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
start_seq = "파이썬"
generated_sentence = generate(model, start_seq)
print("Generated Sentence:", generated_sentence)
