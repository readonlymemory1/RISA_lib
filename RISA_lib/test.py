import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 시계열 데이터 생성 (간단한 예제)
data = list(np.arange(50))+list(np.arange(30))
input_seq_length = 10  # 입력 시퀀스 길이
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
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

# LSTM 기반 시계열 예측 모델 정의
class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMTimeSeriesModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        
        # lstm_out,_ = self.lstm2(lstm_out)
        output= self.fc(lstm_out[:, -1, :])  # 마지막 시간 단계의 출력만 사용
        return output

# 데이터셋과 데이터로더 준비
dataset = TimeSeriesDataset(sequences)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# 모델, 손실 함수, 최적화기 초기화
input_size = 1
hidden_size = 64
output_size = 1
model = LSTMTimeSeriesModel(input_size, hidden_size, output_size)
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_item = []
# 모델 훈련
epochs = 1000
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(-1))
        loss = criterion(outputs, targets.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        if epoch%50==0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
            loss_item.append(loss.item())


plt.plot(loss_item)
plt.show()
# 훈련된 모델로 예측 수행
input_seq = torch.tensor([11, 12, 13, 14, 15], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
predicted_output = model(input_seq)
print("Predicted output:", predicted_output)