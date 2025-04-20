import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者使用SimHei等字体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为乱码

data = pd.read_csv('../processed_data/standardized_data_final.csv')
data_pre = pd.read_csv('../processed_data/standardized_data_pre_final.csv')
X = data.drop(columns=['label'])
feature_names = X.columns.tolist()
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
dpre = xgb.DMatrix(data_pre)

params = {
    'objective': 'binary:logistic',  # 二分类任务
    'eval_metric': 'logloss',        # 评估指标
    'max_depth': 6,                  # 树的最大深度
    'learning_rate': 0.01,            # 学习率
}
num_round = 100

bst = xgb.train(params, dtrain, num_round)
score_dict = bst.get_score(importance_type='weight')

y_pred = bst.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

mypre = bst.predict(dpre)
mypre_binary = (mypre > 0.5).astype(int)

feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
importance_df_xgb = pd.DataFrame([
    {'feature': feature_map.get(k, k), 'importance': v}
    for k, v in score_dict.items()
])

importance_df_xgb['importance_norm'] = importance_df_xgb['importance'] / importance_df_xgb['importance'].sum()

precision_xgb = precision_score(y_test, y_pred_binary)
recall_xgb = recall_score(y_test, y_pred_binary)
f1_xgb = f1_score(y_test, y_pred_binary)
accuracy_xgb = np.sum(y_pred_binary == y_test) / len(y_test)

print(importance_df_xgb)
print(f"XGBoost 模型 - 准确率: {accuracy_xgb}")
print(f"XGBoost 模型 - 查准率: {precision_xgb}")
print(f"XGBoost 模型 - 召回率: {recall_xgb}")
print(f"XGBoost 模型 - F1 分数: {f1_xgb}")

plt.figure(figsize=(12, 7))
sns.barplot(x='importance_norm', y='feature', data=importance_df_xgb, color='royalblue')
plt.xlabel('归一化的重要性', fontsize=16)
plt.ylabel('特征', fontsize=16)
plt.title('XGBoost 特征重要性 (按权重)', fontsize=18, fontweight='bold')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)  # 添加网格线
plt.show()