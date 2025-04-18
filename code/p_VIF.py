import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 读取标准化后的数据
df = pd.read_csv('./processed_data/standardized_data.csv')

# 统计就业状态：0为失业，1为就业
label_counts = df['label'].value_counts()
print("就业状态分布：\n", label_counts)

# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 相关系数矩阵
corr_matrix = df.corr(numeric_only=True)

# 绘制热力图
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("./figure/heatmap.jpg", dpi=500)

# 计算 VIF
X_features = df.drop(columns=['label'])
X_const = add_constant(X_features)

vif_df = pd.DataFrame()
vif_df['Feature'] = X_const.columns
vif_df['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

# 输出 VIF
print(vif_df)

# 保留的特征（排除多重共线性高的）
selected_columns = [
    'birth_month',
    'reg_address_encoded',
    'main_profession_encoded',
    'school_encoded',
    'major_code_encoded',
    'major_name_encoded',
    'sex_性别_enc',
    'nation_民族_enc',
    'marriage_婚姻状态_enc',
    'edu_level_教育程度_enc',
    'politic_政治面貌_enc',
    'religion_宗教信仰_enc',
    'type_人口类型_enc',
    'military_status_兵役状态_enc',
    'is_disability_是否残疾人_enc',
    'is_elder_是否老年人_enc',
    'is_living_alone_是否独居_enc',
    'label'
]

# 正确使用 df 而非未定义的 df_filtered
df_final = df[selected_columns]

# 保存为 CSV
df_final.to_csv("./processed_data/final_processed_data.csv", index=False, encoding='utf-8-sig')
print("数据已保存为 final_processed_data.csv")
