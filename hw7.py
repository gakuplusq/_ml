from micrograd.engine import Value

# 初始化參數 x, y, z（可從任意值開始）
x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

# 學習率
learning_rate = 0.1

# 執行 100 次迭代
for i in range(100):
    # 清空上一輪的梯度
    x.grad = 0
    y.grad = 0
    z.grad = 0

    # 定義目標函數
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 向後傳播計算梯度
    f.backward()

    # 梯度下降（往負梯度方向走）
    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad
    z.data -= learning_rate * z.grad

    # 顯示每一輪的結果
    print(f"Step {i:3d} | f = {f.data:.4f} | x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")
