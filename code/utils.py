import os
import pandas as pd

def load_data(file_path):
    xl = pd.ExcelFile(file_path)

    # 主数据：尝试找到实际表头所在行
    for i in range(10):
        temp_df = xl.parse(sheet_name='数据集', header=i)
        if 'b_aab022' in temp_df.columns:
            header_row = i
            break
    else:
        raise ValueError("未找到包含 'b_aab022' 的表头，请检查表格格式。")

    # 重新读取数据
    data = xl.parse(sheet_name='数据集', header=header_row)
    print("✔ 识别字段成功，实际表头行为第", header_row + 1, "行")
    print("字段名：", data.columns.tolist())

    # 读取行业代码
    code_mapping = xl.parse(sheet_name='行业代码')
    code_mapping = code_mapping.rename(columns={
        '行业代码': '行业分类代码',
        '注释': '行业'
    })

    # 重命名字段并合并行业注释
    data = data.rename(columns={'b_aab022': '行业分类代码'})
    data = pd.merge(data, code_mapping, on='行业分类代码', how='left')

    # 缺失提示
    missing_count = data['行业'].isna().sum()
    print(f"⚠️ 未匹配行业注释的记录数量：{missing_count}")

    return data