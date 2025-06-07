import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------
# 1. 數據準備
# ---------------------
# 輸入：數字 0~9 的 one-hot 向量
inputs = torch.eye(10)

# 輸出：七段顯示器（7 維向量）
targets = torch.tensor([
    [1, 1, 1, 1, 1, 1, 0],  # 0
    [0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 1],  # 2
    [1, 1, 1, 1, 0, 0, 1],  # 3
    [0, 1, 1, 0, 0, 1, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1],  # 5
    [1, 0, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 0, 0, 0, 0],  # 7
    [1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 0, 1, 1],  # 9
], dtype=torch.float32)

# ---------------------
# 2. 建立模型
# ---------------------
class SevenSegmentMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 10)
        self.output = nn.Linear(10, 7)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x

model = SevenSegmentMLP()

# ---------------------
# 3. 訓練設定
# ---------------------
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# ---------------------
# 4. 訓練迴圈
# ---------------------
for epoch in range(3000):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ---------------------
# 5. 結果輸出
# ---------------------
with torch.no_grad():
    predictions = model(inputs).round()
    print("\n預測結果：")
    for i in range(10):
        print(f"{i} -> {predictions[i].int().numpy()}")
