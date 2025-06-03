# 初始化
x, y, z = 任意起點（例如 0, 0, 0）
learning_rate = 0.1
tolerance = 1e-6
max_iter = 1000

for i in range(max_iter):
    # 計算梯度
    grad_x = 2 * x - 2
    grad_y = 2 * y - 4
    grad_z = 2 * z - 6

    # 更新（往梯度反方向走）
    x_new = x - learning_rate * grad_x
    y_new = y - learning_rate * grad_y
    z_new = z - learning_rate * grad_z

    # 收斂判斷
    if abs(x_new - x) < tolerance and abs(y_new - y) < tolerance and abs(z_new - z) < tolerance:
        break

    x, y, z = x_new, y_new, z_new

# 最小點
print(f"Minimum at: ({x}, {y}, {z})")
print(f"Minimum value: {x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8}")
