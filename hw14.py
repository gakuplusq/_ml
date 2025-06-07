import gymnasium as gym  # 或使用 gym==0.26 以上版本
import numpy as np

# 建立環境
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

total_reward = 0
for t in range(500):  # 最長撐到 500 步
    x, x_dot, theta, theta_dot = observation

    # 固定策略：觀察角度和角速度來控制
    if theta < 0:
        action = 0  # 左傾就左推
    else:
        action = 1  # 右傾就右推

    # 更細緻控制：考慮角速度補強穩定性
    # action = 0 if theta + 0.5 * theta_dot < 0 else 1

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

env.close()
print(f"結束！竿子撐了 {total_reward:.0f} 步")
