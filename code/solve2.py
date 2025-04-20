import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

data = pd.read_csv('../processed_data/standardized_data.csv')
data_pre = pd.read_csv('../processed_data/standardized_data_pre.csv')
X = data.drop(columns=['label'])
feature_names = X.columns.tolist()
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

y_pred = bst.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)
mypre = bst.predict(dpre)
mypre_binary = (mypre > 0.5).astype(int)

precision_xgb = precision_score(y_test, y_pred_binary)
recall_xgb = recall_score(y_test, y_pred_binary)
f1_xgb = f1_score(y_test, y_pred_binary)
accuracy_xgb = np.sum(y_pred_binary == y_test) / len(y_test)

print(f"XGBoost模型-准确率: {accuracy_xgb}")
print(f"XGBoost模型-查准率: {precision_xgb}")
print(f"XGBoost模型-召回率: {recall_xgb}")
print(f"XGBoost模型-F1分数: {f1_xgb}")

plot = xgb.plot_importance(bst, importance_type='gain', max_num_features=10) 
plot.set_title("Feature Importance Based on Gain(Top 10 Features)", fontsize=15, fontweight='bold')
plot.set_xlabel("Importance Score (Gain)", fontsize=14)
plot.set_ylabel("Features", fontsize=14)
plot.tick_params(axis='both', which='major', labelsize=12)
plot.set_facecolor('#f4f4f4')
importance_dict = bst.get_score(importance_type='gain')
sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:10]
sorted_importance.reverse()
for p, (feature, score) in zip(plot.patches, sorted_importance):
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plot.annotate(f'{score:.2f}', (x + width, y + height / 2),
                ha='left', va='center', fontsize=10)
plt.show()

