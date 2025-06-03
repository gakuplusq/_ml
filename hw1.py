def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

def gradient(x, y, z):
    return 2 * x - 2, 2 * y - 4, 2 * z - 6

def hill_climb_minimize(start_point, learning_rate=0.1, tolerance=1e-6, max_iter=1000):
    x, y, z = start_point
    for i in range(max_iter):
        grad_x, grad_y, grad_z = gradient(x, y, z)
        x_new = x - learning_rate * grad_x
        y_new = y - learning_rate * grad_y
        z_new = z - learning_rate * grad_z
        if abs(x_new - x) < tolerance and abs(y_new - y) < tolerance and abs(z_new - z) < tolerance:
            break
        x, y, z = x_new, y_new, z_new
    return (x, y, z), f(x, y, z)

start_point = (0.0, 0.0, 0.0)
min_point, min_value = hill_climb_minimize(start_point)

print("最小點：", min_point)
print("最小值：", min_value)
