import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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


def fill_multiple_categorical_columns(df, columns_to_fill):
    df = df.copy()
    for col in columns_to_fill:
        if col not in df.columns:
            continue
        print(f"🔄 正在填补：{col}...")
        not_null_df = df[df[col].notnull()]
        null_df = df[df[col].isnull()]
        if null_df.empty:
            continue

        # 编码器初始化
        label_encoders = {}

        X_train = not_null_df.drop(columns=columns_to_fill)
        y_train = not_null_df[col].astype(str)

        X_test = null_df.drop(columns=columns_to_fill)

        # 对所有类别型变量编码
        for column in X_train.select_dtypes(include='category').columns:
            le = LabelEncoder()
            all_vals = pd.concat([X_train[column], X_test[column]], axis=0).astype(str)
            le.fit(all_vals)
            X_train[column] = le.transform(X_train[column].astype(str))
            X_test[column] = le.transform(X_test[column].astype(str))
            label_encoders[column] = le

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train.select_dtypes(include=[np.number]), y_train)
        y_pred = rf.predict(X_test.select_dtypes(include=[np.number]))

        df.loc[df[col].isnull(), col] = y_pred
        print(f"✅ 填补完成：{col}，填补数量 {len(y_pred)}")

    return df

def advanced_missing_value_processing(df):
    missing_analysis = df.isna().agg(['sum', 'mean']).T
    missing_analysis.columns = ['缺失数量', '缺失比例']
    missing_analysis = missing_analysis.sort_values(by='缺失比例', ascending=False)
    print("缺失值分析报告：\n", missing_analysis.head(10))

    # 删除高缺失率特征（阈值设为70%）
    high_missing_cols = missing_analysis[missing_analysis['缺失比例'] > 0.7].index.tolist()
    if high_missing_cols:
        print(f"🚮 删除高缺失率特征：{high_missing_cols}")
        df = df.drop(columns=high_missing_cols)

    # 🌟 模型填补部分缺失的类别变量
    df = fill_multiple_categorical_columns(df, ['profession', 'c_aac182', 'c_aac183', 'c_aac009', 'c_aac011'])

    # ✅ 毕业年份推算（应放在相关字段填补后）
    if 'c_aac180' in df.columns and 'birthday' in df.columns:
        def infer_grad_year(row):
            if pd.notnull(row.get('graduate_year')):
                return row['graduate_year']
            school = str(row.get('c_aac180', ''))
            birth_date = row.get('birthday', None)
            if pd.isnull(birth_date):
                return np.nan
            birth_year = pd.to_datetime(birth_date, errors='coerce').year
            if pd.isnull(birth_year):
                return np.nan
            # 判断学校类型
            if any(keyword in school for keyword in ['中学', '高中', '职教', '职高', '职工', '技校', '一中', '二中']):
                grad_age = 18
            elif any(keyword in school for keyword in ['大学', '学院', '学校']):
                grad_age = 22
            else:
                grad_age = 20  # 默认值，适用于无法判断的情况
            return birth_year + grad_age

        df['graduate_year'] = df.apply(infer_grad_year, axis=1)
        df['years_since_grad'] = 2025 - df['graduate_year']

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
    assert df['label'].isna().sum() == 0, "标签字段仍存在缺失！"

    return df

import re

# 多对一映射表：每个 key 是目标地区，value 是关键词列表
address_map_multi = {
    "远安县": ["远安"],
    "五峰县": ["五峰"],
    "恩施": ["恩施"],
    "长阳县": ["长阳"],
    "秭归县": ["秭归"],
    "枝江市": ["白洋", "马家店"],
    "宜都市": ["宜都"],
    "宜昌市": ["宜昌", "平云", "夷陵", "城东大道", "发展大道", "黄花", "枝城"],
    "荆门市": ["荆门"],
    "武汉市": ["武汉"],
    "当阳市": ["当阳"],
    "重庆市": ["重庆"],
    "黄冈市": ["麻城", "黄冈"]
}

def extract_city_or_county(address):
    if pd.isna(address):
        return address
    # 多关键词判断
    for region, keywords in address_map_multi.items():
        if any(kw in address for kw in keywords):
            return region
    # 正则匹配提取
    match = re.search(r"([^\s]+省)?\s*([^市\s]+市|[^县\s]+县)", address)
    if match:
        return match.group(2)
    match = re.search(r"([^市\s]+市|[^县\s]+县|[^乡\s]+乡|[^镇\s]+镇|[^区\s]+区)", address)
    if match:
        return match.group(1)
    return address
