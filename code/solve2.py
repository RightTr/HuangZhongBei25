import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者使用SimHei等字体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为乱码

data = pd.read_csv('../processed_data/standardized_data.csv')
data_pre = pd.read_csv('../processed_data/standardized_data_pre.csv')
X = data.drop(columns=['label'])
feature_names = X.columns.tolist()
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
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

feature_translation = {
    'age': '年龄',
    'years_since_grad': '毕业年数',
    'reg_address_enc': '户籍地址',
    'main_profession_encoded': '主职',
    'c_aac011_enc': '文化程度',
    'c_aac180_enc': '学校类型',
    'major_code_encoded': '专业代码编码',
    'major_name_encoded': '专业名称编码',
    'sex_enc': '性别编码',
    'nation_enc': '民族',
    'marriage_enc': '婚姻状态',
    'edu_level_enc': '教育程度',
    'politic_enc': '政治面貌',
    'religion_enc': '宗教信仰',
    'type_enc': '人口类型',
}

importance_dict = bst.get_score(importance_type='gain')
sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:10]

sorted_importance_chinese = [(feature_translation.get(feature, feature), score) for feature, score in sorted_importance]

plt.figure(figsize=(12, 7))
features, scores = zip(*sorted_importance_chinese)
plt.barh(features, scores, color='royalblue', height=0.6)
plt.xlabel('重要性分数 (Gain)', fontsize=16)
plt.ylabel('特征', fontsize=16)
plt.title('基于Gain的特征重要性（前10个特征）', fontsize=18, fontweight='bold')
plt.gca().invert_yaxis()  # 使得特征按重要性从上到下显示

plt.grid(axis='x', linestyle='--', alpha=0.6)  # 仅对x轴添加网格线
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['预测: 0', '预测: 1'], yticklabels=['真实: 0', '真实: 1'])
plt.title("混淆矩阵", fontsize=15, fontweight='bold')
plt.xlabel('预测', fontsize=12)
plt.ylabel('实际', fontsize=12)
plt.show()

cm_df = pd.DataFrame(cm, columns=['预测: 0', '预测: 1'], index=['真实: 0', '真实: 1'])
print("混淆矩阵:\n", cm_df)

