import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import re

def load_data(file_path):
    xl = pd.ExcelFile(file_path)

    for i in range(10):
        temp_df = xl.parse(sheet_name='数据集', header=i)
        if 'b_aab022' in temp_df.columns:
            header_row = i
            break
    else:
        raise ValueError("Error")

    data = xl.parse(sheet_name='数据集', header=header_row)
    print(data.columns.tolist())

    return data

def load_data_pre(file_path):
    xl = pd.ExcelFile(file_path)

    for i in range(10):
        temp_df = xl.parse(sheet_name='预测集', header=i)
        if 'c_aac181' in temp_df.columns:
            header_row = i
            break
    else:
        raise ValueError("Error")

    data = xl.parse(sheet_name='预测集', header=header_row)
    print(data.columns.tolist())

    return data

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
        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            initial_strategy='median'
        )
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # 关键特征验证
    assert df['age'].isna().sum() == 0, "年龄字段仍存在缺失！"

    return df

def extract_city_or_county(address):
    if "远安" in address:
        address = "远安"
    if "五峰" in address:
        address = "五峰"
    if "恩施" in address:
        address = "恩施"
    if "长阳土家族自治县" in address:
        address = "长阳土家族自治县"
    if "秭归" in address:
        address = "秭归"

    match = re.search(r"湖北省\s*([^市\s]+市|[^县\s]+县)", address)
    if match:
        return match.group(1)
    match = re.search(r"([^市\s]+市|[^县\s]+县)", address)
    if match:
        return match.group(1)
    return address