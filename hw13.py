import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. 產生模擬資料
rng = np.random.RandomState(42)

# 正常資料 (200筆，分佈在 2 個中心附近)
X_normal = 0.3 * rng.randn(200, 2)
X_normal = np.r_[X_normal + 2, X_normal - 2]

# 異常資料 (20筆，隨機分佈在較大範圍)
X_outliers = rng.uniform(low=-6, high=6, size=(20, 2))

# 合併資料
X = np.r_[X_normal, X_outliers]

# 2. 建立並訓練 Isolation Forest 模型
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# 3. 預測資料是否異常
y_pred = model.predict(X)  # 正常: 1，異常: -1
scores = model.decision_function(X)  # 分數愈小愈異常

# 4. 可視化結果
plt.figure(figsize=(8, 6))
plt.title("Isolation Forest 異常偵測")
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="coolwarm", edgecolor="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
