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
    missing_analysis = df.isna().agg(['sum', 'mean']).T
    missing_analysis.columns = ['缺失数量', '缺失比例']
    missing_analysis = missing_analysis.sort_values(by='缺失比例', ascending=False)
    print("缺失值分析报告：\n", missing_analysis.head(10))

    # 删除高缺失率特征（阈值设为70%）
    high_missing_cols = missing_analysis[missing_analysis['缺失比例'] > 0.7].index.tolist()
    if high_missing_cols:
        print(f"🚮 删除高缺失率特征：{high_missing_cols}")
        df = df.drop(columns=high_missing_cols)

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

    return df

address_map_multi = {
    "远安县": ["远安", "河口", "鸣凤", "南门村", "嫘祖", "洋坪", "旧县"],
    "五峰土家族自治县": ["五峰", "渔洋关", "长乐坪", "渔关镇"],
    "恩施土家族苗族自治州": ["恩施", "巴东", "建始", "鹤峰", "利川", "宣恩"],
    "长阳土家族自治县": ["长阳", "白氏坪", "渔峡口", "大堰乡", "龙舟坪"],
    "兴山县": ["兴山", "古夫", "昭君镇", "北斗小区", "梅苑小区"],
    "襄阳市": ["襄阳"],
    "仙桃市": ["仙桃"],
    "潜江市": ["潜江"],
    "咸宁市": ["咸宁"],
    "天门市": ["天门"],
    "枣阳市": ["枣阳"],
    "孝感市": ["孝感", "孝昌", "汉川"],
    "鄂州市": ["鄂州"],
    "随州市": ["随州"],
    "十堰市": ["十堰", "郧西"],
    "丹江口市": ["丹江口"],
    "神农架林区": ["神农架"],
    "秭归县": ["秭归", "梅家河", "两河口", "茅坪镇", "陈家冲", "平湖", "水田坝", "泄滩"],
    "枝江市": ["枝江", "白洋", "七星台", "桂溪湖", "龙泉镇", "董市", "马家店", "百里洲", "安福寺", "仙女镇", "余家溪", "公园路", "民主大道", "建材市场", "九畹溪", "马家街道"],
    "宜都市": ["宜都", "园林大道", "解放社区", "枝城", "陆城", "高坝洲", "赤溪河", "红湖", "红春", "松木坪", "中笔社区"],
    "西陵区": ["西陵", "港窑", "北苑路", "珍珠路", "发展大道", "城东大道", "东湖大道", "珠海路", "宜昌市"],
    "伍家岗区": ["伍家岗", "伍临", "东山大道", "夷陵大道", "合益路"],
    "猇亭区": ["猇亭", "下马槽"],
    "夷陵区": ["夷陵", "小溪塔", "罗河路", "分乡", "车站村", "平云", "黄花", "樟村坪", "乐天溪" ,"下堡坪", "邓村乡", "鸦鹊岭", "雾渡河", "三斗坪", "太平溪"],
    "点军区": ["点军"],
    "荆门市": ["荆门", "钟祥", "沙洋"],
    "武汉市": ["武汉"],
    "当阳市": ["当阳", "坝陵"],
    "荆州市": ["荆州", "公安", "监利"],
    "重庆市": ["重庆"],
    "黄石市": ["黄石", "大冶"],
    "黄冈市": ["麻城", "黄冈", "黄梅", "浠水", "蕲春"],
    "黑龙江省": ["黑龙江"],
    "安徽省": ["安徽"],
    "河南省": ["河南"],
    "甘肃省": ["甘肃"],
    "广东省": ["广东", "深圳", "南山区"],
    "湖南省": ["湖南", "长沙"],
    "内蒙古自治区": ["内蒙古"],
    "四川省": ["四川"],
    "云南省": ["云南"],
    "北京市": ["北京"],
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
        return match.group(1)
    match = re.search(r"([^市\s]+市|[^县\s]+县|[^乡\s]+乡|[^镇\s]+镇|[^区\s]+区)", address)
    if match:
        return match.group(1)
    return address

