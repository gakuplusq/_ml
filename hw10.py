import torch
import torch.nn as nn
import torch.optim as optim

# 1. 假設訓練資料：身高(cm) → 體重(kg)，關係大約是 weight = 0.5 * height - 20
x_train = torch.tensor([[160.0], [165.0], [170.0], [175.0], [180.0]])
y_train = torch.tensor([[60.0], [62.5], [65.0], [67.5], [70.0]])

# 2. 建立模型
model = nn.Linear(1, 1)

# 3. 損失函數與優化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 4. 訓練模型
for epoch in range(500):
    # 前向傳播
    y_pred = model(x_train)

    # 計算 loss
    loss = criterion(y_pred, y_train)

    # 反向傳播與更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. 包成 AI 預測函數（模擬呼叫 AI 模型）
def ai_predict(height_cm):
    height_tensor = torch.tensor([[height_cm]])
    with torch.no_grad():
        weight_pred = model(height_tensor)
    return weight_pred.item()

# 6. 呼叫模型做預測
print("📡 呼叫 AI 預測系統")
print("預測 168 cm 的體重為：", round(ai_predict(168), 2), "kg")
print("預測 182 cm 的體重為：", round(ai_predict(182), 2), "kg")
