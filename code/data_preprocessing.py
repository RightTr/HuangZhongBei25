import os
import pandas as pd
from utils import load_data

# 加载数据
data = load_data("../data/附件1.xls")

# 处理缺失值：删除关键列含缺失值的行
key_columns = ['age', 'sex', 'nation', 'edu_level', 'c_aac009']
data = data.dropna(subset=key_columns, how='any')

# 根据特定列删除重复值
data = data.drop_duplicates(subset=['people_id'])

# 自动创建目录
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# 保存清洗后的数据（保留空值）
data.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
print("✅ 已保存清洗后的数据到 data/cleaned_data.csv")
