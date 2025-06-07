import torch

# 初始化變數（需要 requires_grad=True 才能自動求導）
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.1

for step in range(100):
    # 計算函數 f
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 清除之前的梯度
    if x.grad: x.grad.zero_()
    if y.grad: y.grad.zero_()
    if z.grad: z.grad.zero_()

    # 反向傳播計算梯度
    f.backward()

    # 使用梯度做參數更新
    with torch.no_grad():
        x -= learning_rate * x.grad
        y -= learning_rate * y.grad
        z -= learning_rate * z.grad

    print(f"Step {step:3d} | f = {f.item():.4f} | x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")
