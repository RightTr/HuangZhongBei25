import pandas as pd
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from utils import extract_city_or_county, advanced_missing_value_processing

os.makedirs('../processed_data', exist_ok=True)

df_raw = pd.read_csv("../processed_data/data_pre.csv", encoding='utf-8')
df = df_raw.iloc[1:].reset_index(drop=True)
df.replace('\\N', pd.NA, inplace=True)

columns_to_keep = [
    'people_id', 'sex', 'age', 'birthday',
    'nation', 'marriage', 'edu_level', 'politic',
    'reg_address', 'profession', 'religion', 'c_aac009',
    'c_aac011', 'c_aac180', 'c_aac181',
    'type']

df_result = df[columns_to_keep].copy()

# Birthday
df_result['birthday'] = pd.to_datetime(df_result['birthday'], errors='coerce')
df_result['birth_year'] = df_result['birthday'].dt.year
df_result['birth_month'] = df_result['birthday'].dt.month


# Graduate date
df_result['c_aac181'] = pd.to_datetime(df_result['c_aac181'], errors='coerce')
df_result['graduate_year'] = df_result['c_aac181'].dt.year
df_result['years_since_grad'] = 2025 - df_result['graduate_year']

cat_cols = [
    'sex', 'nation', 'marriage', 'edu_level',
    'politic', 'religion', 'c_aac011', 'reg_address', 'c_aac180']

df_result = advanced_missing_value_processing(df_result)

df_result['reg_address'] = df_result['reg_address'].apply(extract_city_or_county)

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

# Agt2Integer
df_result['age'] = pd.to_numeric(df_result['age'], errors='coerce')

# Standardize
scaler = joblib.load('../models/scaler.pkl')
X = df_result[final_features]
X_scaled = scaler.transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)

X_scaled_df.to_csv('../processed_data/standardized_data_pre.csv', index=False, encoding='utf-8-sig')
print("Successfully save to standardized_data_pre.csv")

