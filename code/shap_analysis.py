import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体为微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建图像保存目录
plots_dir = "results/plots"
os.makedirs(plots_dir, exist_ok=True)

# 读取数据及其处理部分
data_path = "results/cleaned_for_analysis.csv"
data = pd.read_csv(data_path)

# 构造目标变量
data['是否在职'] = data['就业状态'].apply(lambda x: 1 if x == '在职' else 0)

# 特征选择
features = ['性别', '学历', '年龄']
X = data[features]
y = data['是否在职']

# 建立预处理器与模型
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), ['性别', '学历']),
    ('num', StandardScaler(), ['年龄'])
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 拟合模型
pipeline.fit(X_train, y_train)

# SHAP分析
onehot = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_features = onehot.get_feature_names_out(['性别', '学历'])
all_features = np.concatenate([cat_features, ['年龄']])

X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)

model = pipeline.named_steps['classifier']
explainer = shap.Explainer(model, X_train_transformed, feature_names=all_features)
shap_values = explainer(X_train_transformed)

# 绘图并保存
shap.summary_plot(shap_values, features=X_train_transformed, feature_names=all_features, show=False)
plt.tight_layout()
plt.savefig(f"{plots_dir}/shap_summary_plot.png", dpi=300)
plt.close()

print("✅ SHAP 特征重要性图已保存：results/plots/shap_summary_plot.png")
