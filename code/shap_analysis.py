import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

plots_dir = "../figure"
os.makedirs(plots_dir, exist_ok=True)

data_path = "../processed_data/standardized_data.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 未找到数据文件：{data_path}")

data['是否在职'] = data['label']

features = [
    'age', 'years_since_grad', 'sex_enc',
    'edu_level_enc', 'marriage_enc',
    'politic_enc', 'cul_level_encoded', 'reg_address_encoded', 'main_profession_encoded',
    'school_encoded', 'major_code_encoded', 'major_name_encoded'
]

X = data[features]
y = data['是否在职']

# 划分训练测试集并拟合逻辑回归模型
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# 生成 SHAP 值
explainer = shap.Explainer(model, X_train, feature_names=features)
shap_values = explainer(X_train)

rename_dict = {
    'age': '年龄',
    'years_since_grad': '毕业年至今',
    'sex_enc': '性别',
    'edu_level_enc': '教育程度',
    'marriage_enc': '婚姻状态',
    'politic_enc': '政治面貌',
    'cul_level_encoded': '文化程度',
    'reg_address_encoded':'户籍地址',
    'main_profession_encoded':'主专业编码',
    'school_encoded':'毕业学校',
    'major_code_encoded':'专业代码',
    'major_name_encoded':'专业名称'
}
readable_features = [rename_dict.get(f, f) for f in features]

shap.summary_plot(shap_values, features=X_train, feature_names=readable_features, show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "shap_summary_plot_filtered.png"), dpi=300)
plt.close()

print("✅ SHAP 特征重要性图已保存：figure/shap_summary_plot_filtered.png")
