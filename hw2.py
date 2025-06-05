import random
import math

# 計算兩點間的歐幾里得距離
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# 計算整個路徑的總距離
def total_distance(path, cities):
    dist = 0
    for i in range(len(path)):
        dist += distance(cities[path[i]], cities[path[(i + 1) % len(path)]])
    return dist

# 產生一個鄰近解：交換兩個城市的位置
def get_neighbor(path):
    a, b = random.sample(range(len(path)), 2)
    neighbor = path[:]
    neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
    return neighbor

# 爬山演算法主程式
def hill_climbing(cities, max_iterations=10000):
    n = len(cities)
    current_path = list(range(n))
    random.shuffle(current_path)
    current_cost = total_distance(current_path, cities)

    for _ in range(max_iterations):
        neighbor = get_neighbor(current_path)
        neighbor_cost = total_distance(neighbor, cities)

        if neighbor_cost < current_cost:
            current_path, current_cost = neighbor, neighbor_cost

    return current_path, current_cost

# 測試：10 個隨機城市
if __name__ == "__main__":
    random.seed(42)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
    
    best_path, best_cost = hill_climbing(cities)
    print("最佳路徑:", best_path)
    print("最短距離:", best_cost)
