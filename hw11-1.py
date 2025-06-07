from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 乳癌資料集（0 = 良性，1 = 惡性）
X, y = load_breast_cancer(return_X_y=True)

# 分割訓練/測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("乳癌分類準確率：", accuracy_score(y_test, y_pred))
