import pandas as pd
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

os.makedirs('../processed_data', exist_ok=True)

df_raw = pd.read_csv("../processed_data/data_pre.csv", encoding='utf-8')

df = df_raw.iloc[2:].reset_index(drop=True)
df.replace('\\N', pd.NA, inplace=True)


columns_to_keep = [
    'people_id', 'name', 'sex', 'birthday', 'age',
    'nation', 'marriage', 'edu_level', 'politic',
    'reg_address', 'profession', 'religion', 'c_aac009',
    'c_aab299', 'c_aac011',
    'c_aac180', 'c_aac181', 'c_aac183',
    'type', 'military_status', 'is_disability',
    'is_teen', 'is_elder', 'change_type',
    'is_living_alone'
]

df_result = df[columns_to_keep].copy()

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

# # 户口所在地
# # df_result['c_aab299_户口所在地区（代码）'] = df_result['c_aab299_户口所在地区（代码）'].astype(str)
# # df_result['hukou_province_code'] = df_result['c_aab299_户口所在地区（代码）'].str[:2]
# # df_result['hukou_city_code'] = df_result['c_aab299_户口所在地区（代码）'].str[2:4]
# # df_result['hukou_county_code'] = df_result['c_aab299_户口所在地区（代码）'].str[4:6]

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

# Agt2Integer
df_result['age'] = pd.to_numeric(df_result['age'], errors='coerce')


# Standardize
X = df_result[final_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)

X_scaled_df.to_csv('../processed_data/standardized_data_pre.csv', index=False, encoding='utf-8-sig')
print("Successfully save to standardized_data_pre.csv")