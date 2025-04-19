import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder

os.makedirs('../processed_data', exist_ok=True)

df_raw = pd.read_csv("../processed_data/data.csv", encoding='utf-8')

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
    'reg_address', 'religion', 'c_aac009',
    'c_aab299', 'c_aac011',
    'c_aac180', 'c_aac181',
    'type', 'military_status', 'is_disability',
    'is_teen', 'is_elder', 'change_type',
    'is_living_alone', 'label'
]

df_result = df[columns_to_keep].copy()

def advanced_missing_value_processing(df):
    # 生成缺失分析报告（建议保留此输出用于验证）
    missing_analysis = df.isna().agg(['sum', 'mean']).T
    missing_analysis.columns = ['缺失数量', '缺失比例']
    missing_analysis = missing_analysis.sort_values(by='缺失比例', ascending=False)
    print("缺失值分析报告：\n", missing_analysis.head(10))

    # 删除高缺失率特征（阈值设为70%）
    high_missing_cols = missing_analysis[missing_analysis['缺失比例'] > 0.7].index.tolist()
    if high_missing_cols:
        print(f"🚮 删除高缺失率特征：{high_missing_cols}")
        df = df.drop(columns=high_missing_cols)

    # 毕业年份推算（根据毕业学校类型判断毕业年龄）
    if 'graduate_school' in df.columns:
        def infer_grad_year(row):
            if pd.notnull(row.get('graduate_year')):
                return row['graduate_year']
            school = str(row.get('c_aac180', ''))
            birth_year = row.get('birth_year', None)
            if pd.isnull(birth_year):
                return np.nan
            # 判断学校类型
            if any(keyword in school for keyword in ['中学', '高中', '职教', '职高', '职工' ,'技校', '一中', '二中']):
                grad_age = 18
            elif any(keyword in school for keyword in ['大学', '学院', '学校']):
                grad_age = 22
            else:
                grad_age = 20  # 默认值，适用于无法判断的情况
            return birth_year + grad_age

        df['graduate_year'] = df.apply(infer_grad_year, axis=1)
        df['years_since_grad'] = 2025 - df['graduate_year']

    # 分类型与数值型差异处理
    # 类别型特征处理（添加'Unknown'类别）
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')

    # 数值型特征处理（MICE算法）
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            initial_strategy='median'
        )
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # 关键特征验证
    assert df['age'].isna().sum() == 0, "年龄字段仍存在缺失！"
    assert df['label'].isna().sum() == 0, "标签字段仍存在缺失！"

    return df

# Birthday
df_result['birthday'] = pd.to_datetime(df_result['birthday'], errors='coerce')
df_result['birth_year'] = df_result['birthday'].dt.year
df_result['birth_month'] = df_result['birthday'].dt.month

# 户籍
le = LabelEncoder()
df_result['reg_address_encoded'] = le.fit_transform(df_result['reg_address'].astype(str))

# Profession
# df_result['main_profession'] = df_result['profession'].astype(str).str.split(',').str[0]
# df_result['main_profession_encoded'] = le.fit_transform(df_result['main_profession'])

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

df_result = advanced_missing_value_processing(df_result)

# Major name
# le_major_name = LabelEncoder()
# df_result['major_name_encoded'] = le_major_name.fit_transform(df_result['c_aac183'].astype(str))

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
    'reg_address_encoded', 'school_encoded',
    ] + [col + '_enc' for col in cat_cols]

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