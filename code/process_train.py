import pandas as pd
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder

os.makedirs('../processed_data', exist_ok=True)

df_raw = pd.read_csv("../processed_data/raw_data.csv", encoding='utf-8')

df = df_raw.iloc[2:].reset_index(drop=True)
df.replace('\\N', pd.NA, inplace=True)

print(df)

today = pd.to_datetime(datetime.today().date())

date_cols = [
    'b_acc031',
    'b_aae031',
    'c_acc028'
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df['label'] = 0
df.loc[df['c_acc0m3'] == '就业', 'label'] = 1
df.loc[df['b_acc031'].notna() & (df['b_aae031'].isna() | (df['b_aae031'] > today)),'label'] = 1
df.loc[df['c_acc028'].notna(), 'label'] = 1

columns_to_keep = [
    'people_id', 'name', 'sex', 'birthday', 'age',
    'nation', 'marriage', 'edu_level', 'politic',
    'reg_address', 'profession', 'religion', 'c_aac009',
    'c_aab299', 'c_aac011',
    'c_aac180', 'c_aac181', 'c_aac183',
    'type', 'military_status', 'is_disability',
    'is_teen', 'is_elder', 'change_type',
    'is_living_alone', 'label'
]

df_result = df[columns_to_keep].copy()

# ## 缺失值处理
# missing_counts = df_result.isna().sum()
# # 计算每列缺失值比例
# missing_ratios = (missing_counts / len(df_result)).round(4)  # 保留4位小数，更直观
# # 合并成一个DataFrame展示
# missing_df = pd.DataFrame({
#     '缺失数量': missing_counts,
#     '缺失比例': missing_ratios
# }).sort_values(by='缺失比例', ascending=False)
# # 打印前几行查看
# print(missing_df)
# # 居住状态这一列删除，其它列填充中位数
# # 2. 填充类别型列的缺失值
# # 找出所有类别型列（object 类型的列）
# categorical_columns = df_result.select_dtypes(include=['object']).columns
# # 对每一列使用众数填充缺失值
# df_result[categorical_columns] = df_result[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))
# # # 打印处理后的缺失值统计
# print(df_result.head(10))

# Birthday
df_result['birthday'] = pd.to_datetime(df_result['birthday'], errors='coerce')
df_result['birth_year'] = df_result['birthday'].dt.year
df_result['birth_month'] = df_result['birthday'].dt.month

# 户籍
le = LabelEncoder()
df_result['reg_address_encoded'] = le.fit_transform(df_result['reg_address'].astype(str))

# Profession
df_result['main_profession'] = df_result['profession'].astype(str).str.split(',').str[0]
df_result['main_profession_encoded'] = le.fit_transform(df_result['main_profession'])

# 户口所在地
# df_result['c_aab299_户口所在地区（代码）'] = df_result['c_aab299_户口所在地区（代码）'].astype(str)
# df_result['hukou_province_code'] = df_result['c_aab299_户口所在地区（代码）'].str[:2]
# df_result['hukou_city_code'] = df_result['c_aab299_户口所在地区（代码）'].str[2:4]
# df_result['hukou_county_code'] = df_result['c_aab299_户口所在地区（代码）'].str[4:6]

# School
df_result['school_encoded'] = le.fit_transform(df_result['c_aac180'].astype(str))

# Graduate date
df_result['c_aac181'] = pd.to_datetime(df_result['c_aac181'], errors='coerce')
df_result['graduate_year'] = df_result['c_aac181'].dt.year
df_result['years_since_grad'] = 2025 - df_result['graduate_year']

# Major name
le_major_name = LabelEncoder()
df_result['major_name_encoded'] = le_major_name.fit_transform(df_result['c_aac183'].astype(str))

cat_cols = [
    'sex', 'nation', 'marriage', 'edu_level',
    'politic', 'religion', 'type', 'military_status',
    'is_disability', 'is_teen', 'is_elder',
    'is_living_alone', 'change_type'
]

for col in cat_cols:
    le = LabelEncoder()
    df_result[col + '_enc'] = le.fit_transform(df_result[col].astype(str))

# Choosed cols
final_features = [
    'age', 'birth_year', 'birth_month', 'graduate_year', 'years_since_grad',
    'reg_address_encoded', 'main_profession_encoded', 'school_encoded',
    'major_name_encoded'] + [col + '_enc' for col in cat_cols]

# Count duplicate ids
dup_counts = df_result['people_id'].value_counts()
duplicate_ids = dup_counts[dup_counts > 1].index
print(f"Duplicate_ids_count: {len(duplicate_ids)}")

# Remove duplicate ids rows
df_result = df_result[~df_result['people_id'].isin(duplicate_ids)].reset_index(drop=True)
print(f"Removed result: {len(df_result)}")

# Save Non_Standardized_Data
df_result.to_csv('../processed_data/non_standardized_data.csv', index=False, encoding='utf-8-sig')
print("Successfully save to non_standardized_data.csv")

# Agt2Integer
df_result['age'] = pd.to_numeric(df_result['age'], errors='coerce')

df_model = df_result[final_features + ['label']]

# Standardize
from sklearn.preprocessing import StandardScaler
X = df_model[final_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)
X_scaled_df['label'] = df_model['label'].values

X_scaled_df.to_csv('../processed_data/standardized_data.csv', index=False, encoding='utf-8-sig')
print("Successfully save to standardized_data.csv")