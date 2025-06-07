import torch
import torch.nn as nn
import torch.optim as optim

# 1. 建立訓練資料 (假設是 y = 3x + 2)
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

# 2. 建立模型（線性回歸）
model = nn.Linear(in_features=1, out_features=1)

# 3. 定義損失函數與優化器
criterion = nn.MSELoss()  # 均方誤差
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 訓練模型
for epoch in range(500):
    # 預測
    y_pred = model(x_train)

    # 計算損失
    loss = criterion(y_pred, y_train)

    # 清空梯度
    optimizer.zero_grad()

    # 反向傳播
    loss.backward()

    # 更新參數
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

# 5. 查看訓練後的參數
w = model.weight.item()
b = model.bias.item()
print(f"\n學到的模型：y = {w:.2f}x + {b:.2f}")
