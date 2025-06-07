from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 產生 3 群資料（模擬分群場景）
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 建立並訓練 KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# 顯示分群結果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title("KMeans 分群結果")
plt.show()
