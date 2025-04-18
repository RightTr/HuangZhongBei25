import pandas as pd
from datetime import datetime

# 读取原始数据
df_raw = pd.read_csv("../code/data/data.csv", encoding='utf-8')  # 视情况可能需要换 encoding
# 合并前两行作为真正的表头（index=0是拼音，1是中文）
new_columns = df_raw.iloc[0].fillna('') + "_" + df_raw.iloc[1].fillna('')
df_raw.columns = new_columns
# 去掉前两行，只保留实际数据
df = df_raw.iloc[2:].reset_index(drop=True)
# 将缺失值 \N 替换为 NaN
df.replace('\\N', pd.NA, inplace=True)
# 查看清洗后的前几行
print(df)

# 当前日期
today = pd.to_datetime(datetime.today().date())
# 转换相关日期字段为 datetime 格式
date_cols = [
    'b_acc031_就业时间',
    'b_aae031_合同终止日期',
    'c_acc028_失业注销时间'
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
# 初始化标签，默认认为是“失业”状态
df['label'] = 0
# 条件1：登记为就业
df.loc[df['c_acc0m3_登记就失业状态'] == '就业', 'label'] = 1
# 条件2：有“就业时间”，且“合同终止日期”为空或晚于今天
df.loc[
    df['b_acc031_就业时间'].notna() & (
        df['b_aae031_合同终止日期'].isna() |
        (df['b_aae031_合同终止日期'] > today)
    ),
    'label'
] = 1

# 条件3：有“失业注销时间” → 就业
df.loc[df['c_acc028_失业注销时间'].notna(), 'label'] = 1

# 可选：查看标注结果
print(df[['name_姓名', 'b_acc031_就业时间', 'b_aae031_合同终止日期', 'c_acc028_失业注销时间', 'c_acc0m3_登记就失业状态', 'label']].head())

## 剔除无关变量
# 设置保留字段
columns_to_keep = [
    'people_id_人员编11号', 'name_姓名', 'sex_性别', 'birthday_生日', 'age_年龄',
    'nation_民族', 'marriage_婚姻状态', 'edu_level_教育程度', 'politic_政治面貌',
    'reg_address_户籍地址', 'profession_专业', 'religion_宗教信仰', 'c_aac009_户口性质',
    'c_aab299_户口所在地区（代码）', 'c_aac010_户口所在地区（名称）', 'c_aac011_文化程度',
    'c_aac180_毕业学校', 'c_aac181_毕业日期', 'c_aac182_所学专业代码', 'c_aac183_所学专业名称',
    'type_人口类型', 'military_status_兵役状态', 'is_disability_是否残疾人',
    'is_teen_是否青少年', 'is_elder_是否老年人', 'change_type_变动类型',
    'is_living_alone_是否独居', 'live_status_居住状态', 'label'
]

# 筛选字段副本
df_result = df[columns_to_keep].copy()
print(df_result)

## 缺失值处理
missing_counts = df_result.isna().sum()
# 计算每列缺失值比例
missing_ratios = (missing_counts / len(df_result)).round(4)  # 保留4位小数，更直观
# 合并成一个DataFrame展示
missing_df = pd.DataFrame({
    '缺失数量': missing_counts,
    '缺失比例': missing_ratios
}).sort_values(by='缺失比例', ascending=False)
# 打印前几行查看
print(missing_df)
# 居住状态这一列删除，其它列填充中位数
# 1. 删除缺失值过多的列
df_result = df_result.drop(columns=['live_status_居住状态'])
# 2. 填充类别型列的缺失值
# 找出所有类别型列（object 类型的列）
categorical_columns = df_result.select_dtypes(include=['object']).columns
# 对每一列使用众数填充缺失值
df_result[categorical_columns] = df_result[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))
# # 打印处理后的缺失值统计
print(df_result.head(10))


## 3. 特征转换与派生
# 生日
# 将生日列转换为 datetime 类型
df_result['birthday_生日'] = pd.to_datetime(df_result['birthday_生日'], errors='coerce')
# 新增出生年份、月份特征
df_result['birth_year'] = df_result['birthday_生日'].dt.year
df_result['birth_month'] = df_result['birthday_生日'].dt.month

# 户籍地址
df_result['province'] = df_result['reg_address_户籍地址'].str.extract(r'^(.*?省)')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_result['reg_address_encoded'] = le.fit_transform(df_result['reg_address_户籍地址'].astype(str))

# 提取主专业代码
df_result['main_profession'] = df_result['profession_专业'].astype(str).str.split(',').str[0]
# 编码为数字标签
df_result['main_profession_encoded'] = le.fit_transform(df_result['main_profession'])

# 户口所在地
df_result['c_aab299_户口所在地区（代码）'] = df_result['c_aab299_户口所在地区（代码）'].astype(str)
# 提取省、市、县代码
df_result['hukou_province_code'] = df_result['c_aab299_户口所在地区（代码）'].str[:2]
df_result['hukou_city_code'] = df_result['c_aab299_户口所在地区（代码）'].str[2:4]
df_result['hukou_county_code'] = df_result['c_aab299_户口所在地区（代码）'].str[4:6]

# 毕业学校
df_result['school_encoded'] = le.fit_transform(df_result['c_aac180_毕业学校'].astype(str))

# 毕业日期
# 转换为 datetime 类型
df_result['c_aac181_毕业日期'] = pd.to_datetime(df_result['c_aac181_毕业日期'], errors='coerce')
# 提取毕业年份
df_result['graduate_year'] = df_result['c_aac181_毕业日期'].dt.year
# 计算距今年数（以 2025 年为基准）
df_result['years_since_grad'] = 2025 - df_result['graduate_year']

# 专业代码
from sklearn.preprocessing import LabelEncoder
le_major_code = LabelEncoder()
df_result['major_code_encoded'] = le_major_code.fit_transform(df_result['c_aac182_所学专业代码'].astype(str))

# 专业名称
le_major_name = LabelEncoder()
df_result['major_name_encoded'] = le_major_name.fit_transform(df_result['c_aac183_所学专业名称'].astype(str))

# 输出处理后的数据
print(df_result.dtypes)

# 把其它特征转换成数值型
from sklearn.preprocessing import LabelEncoder

# 需要编码的类别型字段
cat_cols = [
    'sex_性别', 'nation_民族', 'marriage_婚姻状态', 'edu_level_教育程度',
    'politic_政治面貌', 'religion_宗教信仰', 'type_人口类型', 'military_status_兵役状态',
    'is_disability_是否残疾人', 'is_teen_是否青少年', 'is_elder_是否老年人',
    'is_living_alone_是否独居', 'change_type_变动类型'
]
# 对所有类别型列进行 LabelEncoding
for col in cat_cols:
    le = LabelEncoder()
    df_result[col + '_enc'] = le.fit_transform(df_result[col].astype(str))

# 选取最终用于建模的字段
final_features = [
                     # 数值型与衍生信息
                     'age_年龄', 'birth_year', 'birth_month', 'graduate_year', 'years_since_grad',

                     # 已编码的字段
                     'reg_address_encoded', 'main_profession_encoded', 'school_encoded',
                     'major_code_encoded', 'major_name_encoded',

                     # 刚刚LabelEncode的类别变量
                 ] + [col + '_enc' for col in cat_cols]

# 将年龄转为数值型（int）
df_result['age_年龄'] = pd.to_numeric(df_result['age_年龄'], errors='coerce')

# 取建模用数据子集
df_model = df_result[final_features + ['label']]
print(df_model)

from sklearn.preprocessing import StandardScaler

# 最终建模用的特征（假设你之前已经保存在 final_features 中）
X = df_model[final_features]

# 初始化标准化器
scaler = StandardScaler()

# 拟合并变换
X_scaled = scaler.fit_transform(X)

# 转换为DataFrame并保留列名
X_scaled_df = pd.DataFrame(X_scaled, columns=final_features)

# 如果需要添加标签（label）列
X_scaled_df['label'] = df_model['label'].values

# 查看标准化后的结果
print(X_scaled_df.head())
# 保存为 CSV 文件
X_scaled_df.to_csv('./processed_data/standardized_data.csv', index=False, encoding='utf-8-sig')
