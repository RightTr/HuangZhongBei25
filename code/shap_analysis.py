import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 设置字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建图像保存目录
plots_dir = "../figure"
os.makedirs(plots_dir, exist_ok=True)

# 读取数据
data_path = "../processed_data/standardized_data.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 未找到数据文件：{data_path}")

# 构造目标变量
data['是否在职'] = data['label']

# 精选特征（已标准化处理）
features = [
    'age', 'years_since_grad', 'sex_enc',
    'edu_level_enc', 'marriage_enc',
    'politic_enc','c_aac011_enc'
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

# 中文别名映射
rename_dict = {
    'age': '年龄',
    'years_since_grad': '毕业年至今',
    'sex_enc': '性别',
    'edu_level_enc': '教育程度',
    'marriage_enc': '婚姻状态',
    'politic_enc': '政治面貌',
    'c_aac011_enc': '文化程度'
}
readable_features = [rename_dict.get(f, f) for f in features]

# 绘制并保存 SHAP 总结图
shap.summary_plot(shap_values, features=X_train, feature_names=readable_features, show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "shap_summary_plot_filtered.png"), dpi=300)
plt.close()

print("✅ SHAP 特征重要性图已保存：figure/shap_summary_plot_filtered.png")
