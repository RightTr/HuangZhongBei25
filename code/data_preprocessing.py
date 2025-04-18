import os
import pandas as pd
from utils import load_data

# 加载数据
data = load_data("../data/附件1.xls")

# 处理缺失值：删除关键列含缺失值的行
key_columns = ['birthday', 'age', 'sex', 'nation', 'marriage', 'edu_level', 'profession', 'military_status']
data = data.dropna(subset=key_columns, how='any')

# 处理其他列的缺失值：数值型用中位数填充，分类变量用众数填充
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# 对其他列进行缺失值填充
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# 删除重复值
data = data.drop_duplicates()

# 自动创建目录
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# 保存清洗后的数据（含行业名称）
data.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
print("✅ 已保存清洗后的数据到 data/cleaned_data.csv")
