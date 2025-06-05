import random
import math
import matplotlib.pyplot as plt

# 計算兩城市的距離
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# 計算整個巡迴的長度
def total_distance(path, cities):
    dist = 0
    for i in range(len(path)):
        dist += distance(cities[path[i]], cities[path[(i + 1) % len(path)]])
    return dist

# 產生鄰居解：隨機交換兩個城市
def get_neighbor(path):
    a, b = random.sample(range(len(path)), 2)
    neighbor = path[:]
    neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
    return neighbor

# 計算 "高度"（越高表示越短路徑）
def height(path, cities):
    return 1 / total_distance(path, cities)

# 畫出城市與巡迴路徑
def plot_path(cities, path, title="TSP Path"):
    ordered = [cities[i] for i in path] + [cities[path[0]]]
    x, y = zip(*ordered)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-r')
    plt.title(title)
    for idx, (cx, cy) in enumerate(cities):
        plt.text(cx + 1, cy + 1, str(idx), fontsize=12)
    plt.grid(True)
    plt.show()

# 爬山演算法主程式
def hill_climbing(cities, max_iterations=10000):
    n = len(cities)
    current_path = list(range(n))
    random.shuffle(current_path)
    current_height = height(current_path, cities)

    print("初始長度:", total_distance(current_path, cities))

    for _ in range(max_iterations):
        neighbor = get_neighbor(current_path)
        neighbor_height = height(neighbor, cities)

        if neighbor_height > current_height:
            current_path = neighbor
            current_height = neighbor_height

    final_dist = total_distance(current_path, cities)
    print("最終長度:", final_dist)
    return current_path

# 測試
if __name__ == "__main__":
    random.seed(42)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
    best_path = hill_climbing(cities)
    plot_path(cities, best_path, title="Hill Climbing Solution")
