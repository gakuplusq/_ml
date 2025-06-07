from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加州房價資料集
data = fetch_california_housing()
X, y = data.data, data.target

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立回歸模型
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("房價預測 MSE：", mean_squared_error(y_test, y_pred))
