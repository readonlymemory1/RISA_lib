import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import re
from Sam_train import train_tool
from nltk.tokenize import sent_tokenize

# 시계열 데이터 생성 (간단한 예제)
sentence_data = '''Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.

Though wise men at their end know dark is right,
Because their words had forked no lightning they
Do not go gentle into that good night.

Good men, the last wave by, crying how bright
Their frail deeds might have danced in a green bay,
Rage, rage against the dying of the light.

Wild men who caught and sang the sun in flight,
And learn, too late, they grieved it on its way,
Do not go gentle into that good night.

Grave men, near death, who see with blinding sight
Blind eyes could blaze like meteors and be gay,
Rage, rage against the dying of the light.

And you, my father, there on the sad height,
Curse, bless, me now with your fierce tears, I pray.
Do not go gentle into that good night.
Rage, rage against the dying of the light.
'''

sentence_data = re.sub(r'[^a-zA-Z0-9 ]', ' ', sentence_data).lower()
tokenize = sent_tokenize(sentence_data)
# 단어 인코더와 디코더 정의
class TrainTool:
    def __init__(self, data):
        self.data = data.split()
        self.word2index = {word: i for i, word in enumerate(set(self.data))}
        self.index2word = {i: word for word, i in self.word2index.items()}

    def i2w(self):
        return self.index2word

    def w2i(self):
        return self.word2index

tool = TrainTool(sentence_data)
tools = train_tool(sentence_data)
index2word = tools.i2w()
word2index = tools.w2i()

def encoder(x):
    data_split = x.split()
    data = []
    for i in range(len(data_split)):
        data.append(word2index[data_split[i]])
    return data

def decoder(x):
    return index2word[x]

data = encoder(sentence_data)

input_seq_length = 3  # 입력 시퀀스 길이

output_seq_length = 1  # 출력 시퀀스 길이

# 데이터를 입력 시퀀스와 출력 시퀀스로 변환
def create_sequences(data, input_length, output_length):
    sequences = []
    for i in range(len(data) - input_length - output_length + 1):
        input_seq = data[i:i+input_length]
        output_seq = data[i+input_length:i+input_length+output_length]
        sequences.append((input_seq, output_seq))
    return sequences

sequences = create_sequences(data, input_seq_length, output_seq_length)

# 데이터셋 클래스 정의
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, output_seq = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_seq, dtype=torch.long)

# LSTM 기반 시계열 예측 모델 정의
class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMTimeSeriesModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        output = self.fc1(lstm_out[:, -1, :])  # 마지막 시간 단계의 출력만 사용
        return output

# 데이터셋과 데이터로더 준비
dataset = TimeSeriesDataset(sequences)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# 모델, 손실 함수, 최적화기 초기화
input_size = 1
hidden_size = 64
output_size = len(word2index)  # 출력 크기는 고유한 단어의 수
model = LSTMTimeSeriesModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_item = []

# 모델 훈련
epochs = 100
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        inputs = inputs.unsqueeze(-1).float()
        targets = targets.squeeze()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
        loss_item.append(loss.item())

plt.plot(loss_item)
plt.show()

# 훈련된 모델로 예측 수행
a = ["do", "not", "go"]
for i in range(50):

    print(a)
    encoding = tools.encoder(" ".join(a))
    print(tools.using_for_decoder(encoding))
    input_seq = torch.tensor(encoding[len(a)-5:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    predicted_output = model(input_seq)
    predicted_word_index = torch.argmax(predicted_output, dim=-1).item()
    print("Predicted output:", tools.decoder(predicted_word_index))
    a.append(tools.decoder(predicted_word_index))
print(' '.join(a))
