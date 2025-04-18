import pandas as pd
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from utils import advanced_missing_value_processing

os.makedirs('../processed_data', exist_ok=True)

df_raw = pd.read_csv("../processed_data/data.csv", encoding='utf-8')

df = df_raw.iloc[2:].reset_index(drop=True)
df.replace('\\N', pd.NA, inplace=True)

print(df)

today = pd.to_datetime(datetime.today().date())

date_cols = [
    'b_acc031', # 就业时间
    'b_aae031', # 合同终止日期
    'c_acc028' # 失业注销时间
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df['label'] = 0
df.loc[df['c_acc0m3'] == '就业', 'label'] = 1
df.loc[df['b_acc031'].notna() & (df['b_aae031'].isna() | (df['b_aae031'] > today)),'label'] = 1
df.loc[df['c_acc028'].notna(), 'label'] = 1

# c_aac009: 户口性质
# c_aab299: 户口所在地区（代码）
# c_aac011: 文化程度
# c_aac180: 毕业学校
# c_aac181: 毕业日期
# c_aac183: 所学专业名称
columns_to_keep = [
    'people_id', 'sex', 'age', 'birthday',
    'nation', 'marriage', 'edu_level', 'politic',
    'reg_address', 'profession', 'religion', 'c_aac009',
    'c_aac011', 'c_aac180', 'c_aac181',
    'type', 'label'
]

df_result = df[columns_to_keep].copy()

# Graduate date
df_result['c_aac181'] = pd.to_datetime(df_result['c_aac181'], errors='coerce')
df_result['graduate_year'] = df_result['c_aac181'].dt.year
df_result['years_since_grad'] = 2025 - df_result['graduate_year']

df_result = advanced_missing_value_processing(df_result)

cat_cols = [
    'sex', 'nation', 'marriage', 'edu_level',
    'politic', 'religion', 'type', 'c_aac011']

for col in cat_cols:
    le = LabelEncoder()
    df_result[col + '_enc'] = le.fit_transform(df_result[col].astype(str))

# Choosed cols
final_features = [
    'age', 'years_since_grad'] + [col + '_enc' for col in cat_cols]

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
X = df_model[final_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)
X_scaled_df['label'] = df_model['label'].values

X_scaled_df.to_csv('../processed_data/standardized_data.csv', index=False, encoding='utf-8-sig')
print("Successfully save to standardized_data.csv")
