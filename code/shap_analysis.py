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
plots_dir = "figure"
os.makedirs(plots_dir, exist_ok=True)

# 读取数据
data_path = "processed_data/standardized_data.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 未找到数据文件：{data_path}")

# 构造目标变量
data['是否在职'] = data['label']

# 精选特征（已标准化处理）
features = [
    'age_年龄', 'years_since_grad', 'sex_性别_enc',
    'edu_level_教育程度_enc', 'marriage_婚姻状态_enc',
    'politic_政治面貌_enc', 'is_disability_是否残疾人_enc',
    'is_elder_是否老年人_enc', 'is_teen_是否青少年_enc',
    'is_living_alone_是否独居_enc'
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
    'age_年龄': '年龄',
    'years_since_grad': '毕业年距今',
    'sex_性别_enc': '性别',
    'edu_level_教育程度_enc': '教育程度',
    'marriage_婚姻状态_enc': '婚姻状态',
    'politic_政治面貌_enc': '政治面貌',
    'is_disability_是否残疾人_enc': '是否残疾人',
    'is_elder_是否老年人_enc': '是否老年人',
    'is_teen_是否青少年_enc': '是否青少年',
    'is_living_alone_是否独居_enc': '是否独居'
}
readable_features = [rename_dict.get(f, f) for f in features]

# 绘制并保存 SHAP 总结图
shap.summary_plot(shap_values, features=X_train, feature_names=readable_features, show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "shap_summary_plot_filtered.png"), dpi=300)
plt.close()

print("✅ SHAP 特征重要性图已保存：figure/shap_summary_plot_filtered.png")
