import pandas as pd
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from utils import advanced_missing_value_processing, extract_city_or_county
import joblib 

os.makedirs('../processed_data', exist_ok=True)

df_raw = pd.read_csv("../processed_data/data.csv", encoding='utf-8')
df = df_raw.iloc[1:].reset_index(drop=True)
df.replace('\\N', pd.NA, inplace=True)

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
    'c_aac011', 'c_aac180', 'c_aac181', 'type', 'label']

# 创建高新技术产业增加值数据
data_high_tech = {
    '地区名': ['宜昌市', '宜都市', '枝江市', '当阳市', '远安县', '兴山县', '秭归县', '长阳土家族自治县', '五峰自治县', '夷陵区', '西陵区', '伍家岗区', '点军区', '猇亭区', '宜昌高新区'],
    2017: [406.41, 61.12, 43.11, 37.07, 55.94, 4.88, 7.07, 7.77, 2.78, 30.42, 83.92, 9.65, 4.83, 34.42, 23.92],
    2018: [472.49, 83.69, 61.24, 55.19, 20.4, 8.11, 10.33, 8.89, 4.12, 44.32, 81.13, 15.57, 9.25, 42.81, 31.27],
    2019: [644.65, 119.25, 85.99, 103.57, 14.22, 9.62, 22.01, 17.2, 5.97, 35.5, 92.26, 25.25, 10.36, 53.14, 51.28],
    2020: [649.61, 114.78, 106.5, 81.69, 25.41, 8.08, 4.18, 19.23, 6.71, 42.47, 99.44, 25, 6.84, 56.08, 57.69],
    2021: [874.15, 152.06, 161, 87.8, 33.63, 12.56, 14.12, 26.08, 10.75, 61.14, 123.15, 35.47, 10.59, 76.64, 75.69],
    2022: [1253.7, 243.18, 234.26, 111.05, 49.66, 18.97, 22.88, 26.53, 14.5, 104.83, 148.51, 59.03, 12.43, 126.14, 94.55]
}


df_high_tech = pd.DataFrame(data_high_tech)
df_high_tech.set_index('地区名', inplace=True) # 设置地区名字为索引
years = [col for col in df_high_tech.columns if isinstance(col, int)]
growth_rate_df = df_high_tech.copy()
aagr = (df_high_tech[years].iloc[:, 1:].values - df_high_tech[years].iloc[:, :-1].values) / df_high_tech[years].iloc[:, :-1].values
aagr_mean = aagr.mean(axis=1)
df_high_tech['Average Mean'] = aagr_mean

df_result = df[columns_to_keep].copy()

# Graduate date
df_result['c_aac181'] = pd.to_datetime(df_result['c_aac181'], errors='coerce')
df_result['graduate_year'] = df_result['c_aac181'].dt.year
df_result = advanced_missing_value_processing(df_result)
df_result['years_since_grad'] = 2025 - df_result['graduate_year']
df_result['graduate_year'] = df_result['c_aac181'].dt.year

df_result['reg_address'] = df_result['reg_address'].apply(extract_city_or_county)
df_result['reg_address'].to_csv('../temp/reg.csv', index=False, encoding='utf-8-sig')

# get mean
def get_value(row):
    region = row['reg_address']
    try:
        return df_high_tech.loc[region, 'Average Mean']
    except KeyError:
        return None

df_result['high_tech'] = df_result.apply(get_value, axis=1)

nan_count = df_result['high_tech'].isna().sum()
print(f"high_tech isnan: {nan_count}")


# remove nan
df_result = df_result.dropna(subset=['high_tech'])

cat_cols = [
    'sex', 'nation', 'marriage', 'edu_level',
    'politic', 'religion', 'c_aac011', 'reg_address', 'high_tech']

for col in cat_cols:
    le = LabelEncoder()
    df_result[col + '_enc'] = le.fit_transform(df_result[col].astype(str))

# Choosed cols
final_features = ['age', 'years_since_grad'] + [col + '_enc' for col in cat_cols]

# Count duplicate ids
dup_counts = df_result['people_id'].value_counts()
duplicate_ids = dup_counts[dup_counts > 1].index
print(f"Duplicate_ids_count: {len(duplicate_ids)}")

# Remove duplicate ids rows
df_result = df_result[~df_result['people_id'].isin(duplicate_ids)].reset_index(drop=True)
print(f"Removed result: {len(df_result)}")

# Agt2Integer
df_result['age'] = pd.to_numeric(df_result['age'], errors='coerce')

df_model = df_result[final_features + ['label']]

# Standardize
X = df_model[final_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, '../models/scaler_final.pkl')
print("Successfully save to scaler_final.pkl")

X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)
X_scaled_df['label'] = df_model['label'].values

X_scaled_df.to_csv('../processed_data/standardized_data_final.csv', index=False, encoding='utf-8-sig')
print("Successfully save to standardized_data_final.csv")
