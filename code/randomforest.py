import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取预处理后的数据
df = pd.read_csv('../processed_data/final_processed_data.csv')

# 目标变量 (label) 和特征 (features)
X = df.drop(columns=['label'])
print(X.shape)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 初始化随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 输出结果
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"查准率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 值: {f1:.4f}")

# 绘制特征重要性图
importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]  # 从大到小排序

# 提取特征名称
feature_names = X.columns

# 绘制条形图
plt.figure(figsize=(10, 8))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title('Feature Importance')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig("figure/feature_importance.jpg", dpi=500)


# 保存训练好的模型
joblib.dump(rf_model, 'random_forest_model.pkl')
print(" 模型已保存为 'random_forest_model.pkl'")

## 加载模型直接预测预测集数据
predict_data = pd.read_csv("./processed_data/predict_scaled_data.csv")
y_pred = rf_model.predict(predict_data)

# 保存结果
df_result = predict_data.copy()
df_result['预测结果'] = y_pred
df_result.to_csv('./processed_data/predict_result1.csv', index=False, encoding='utf-8-sig')
print("预测完成，结果保存为 ./processed_data/predict_result.csv")