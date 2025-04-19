import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

feature_rename_map = {
    'age': '年龄',
    'sex_enc': '性别',
    'nation_enc': '民族',
    'birth_month':'出生月份',
    'birth_year':'出生年份',
    'marriage_enc': '婚姻状况',
    'edu_level_enc': '教育程度',
    'politic_enc': '政治面貌',
    'religion_enc': '宗教信仰',
    'cul_level_encoded': '文化程度',
    'military_status_enc':'兵役状态',
    'is_disability_enc': '是否残疾',
    'is_elder_enc':'是否老年',
    'is_teen_enc': '是否青少年',
    'type_enc':'人口类型',
    'years_since_grad': '毕业年距',
    'reg_address_encoded': '户籍地址',
    'main_profession_encoded': '主专业编码',
    'school_encoded': '毕业学校',
    'major_code_encoded': '专业代码',
    'major_name_encoded': '专业名称',
    'label': '是否就业'
}

def load_and_validate_data(path):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        print(f"成功加载数据，维度：{df.shape}")

        assert 'label' in df.columns, "数据中缺少必需的label列"
        assert df.shape[0] > 100, "数据量过少（行数<100），请检查数据文件"

        return df
    except Exception as e:
        print(f" 数据加载失败：{str(e)}")
        raise


def generate_correlation_heatmap(df):
    # 计算相关系数
    corr_matrix = df.corr(numeric_only=True)

    corr_matrix = corr_matrix.rename(index=feature_rename_map, columns=feature_rename_map)

    # 创建mask矩阵
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    plt.figure(figsize=(20, 18))
    cmap = sns.diverging_palette(240, 10, s=80, l=50, as_cmap=True)

    # 绘制相关系数矩阵（热力图）
    ax = sns.heatmap(
        corr_matrix.round(2),
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "相关系数"},
        annot_kws={"size": 10, "color": "#2d2d2d"}
    )

    ax.set_title("特征相关系数矩阵分析\n",
                 fontsize=24,
                 fontweight='bold',
                 pad=25)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # 保存输出
    output_path = "../figure/correlation_matrix.jpg"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\n相关系数矩阵图已保存至：{output_path}")

def perform_vif_analysis(df, vif_threshold=10):
    X = df.drop(columns=['label'])
    X_const = add_constant(X)

    vif_results = []
    for i in range(X_const.shape[1]):
        feature_name = X_const.columns[i]
        try:
            vif = variance_inflation_factor(X_const.values, i)
        except Exception as e:
            print(f" 计算特征 '{feature_name}' 时发生错误：{str(e)}")
            vif = np.inf
        vif_results.append(vif)

    vif_df = pd.DataFrame({
        '特征名称': X_const.columns,
        'VIF值': vif_results,
        '共线性评价': pd.cut(
            vif_results,
            bins=[0, 5, 10, 50, np.inf],
            labels=["无共线性", "轻度共线性", "显著共线性", "严重共线性"],
            right=False
        )
    }).sort_values('VIF值', ascending=False)

    # 筛选建议
    high_vif_features = vif_df[vif_df['VIF值'] > vif_threshold]['特征名称'].tolist()
    if high_vif_features:
        print("\n高共线性特征建议：")
        print(" | ".join(high_vif_features))

    return vif_df

def generate_markdown_report(report_data):
    try:
        md_content = f"""
# 数据特征分析报告
**生成时间**：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 目标变量分布

{report_data['label_dist']}

## 2. 特征共线性分析（VIF）

{report_data['vif_table'].to_markdown(index=False)}

**共线性说明**：
- 严重共线性（VIF>50）：红色标记特征
- 显著共线性（VIF>10）：橙色标记特征
- 轻度共线性（VIF>5）：黄色标记特征

## 3. 特征相关性热力图
![相关系数矩阵](./figure/correlation_matrix.jpg)

## 4. 分析建议

### 数据质量问题
{report_data['quality_issues']}

### 特征工程建议
1. **共线性处理**：
   - 删除特征：`{', '.join(report_data['high_vif_features'])}`
   - 使用PCA降维（保留85%方差）
2. **特征优化**：
   - 离散化处理：`birth_month` 转换为季度特征
   - 嵌入编码：`main_profession_encoded` 使用FastText编码

### 建模建议
- 样本重采样（SMOTE过采样）
- 使用LightGBM内置类别特征处理
- 添加特征交互项：`school_encoded × major_code_encoded`
        """

        report_path = "../processed_data/reports/analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"\nMarkdown分析报告已生成：{report_path}")
        return report_path

    except Exception as e:
        print(f"\n报告生成失败：{str(e)}")
        raise

if __name__ == "__main__":
    try:
        df = load_and_validate_data('../processed_data/standardized_data.csv')

        report_data = {}

        label_dist = df['label'].value_counts().reset_index()
        label_dist.columns = ['就业状态', '数量']
        report_data['label_dist'] = label_dist.to_markdown(index=False)

        vif_report = perform_vif_analysis(df)

        vif_report['特征名称'] = vif_report['特征名称'].replace(feature_rename_map)

        vif_report['特征名称'] = vif_report.apply(
            lambda x: f"<span style='color:red'>{x['特征名称']}</span>"
            if x['VIF值'] > 50 else (
                f"<span style='color:orange'>{x['特征名称']}</span>"
                if x['VIF值'] > 10 else x['特征名称']
            ), axis=1
        )

        report_data['vif_table'] = vif_report

        high_vif = vif_report[vif_report['VIF值'] > 10]
        report_data['high_vif_features'] = high_vif['特征名称'].tolist()
        report_data['quality_issues'] = "- 检测到 {} 个高共线性特征（VIF>10）".format(
            len(report_data['high_vif_features'])
        )

        generate_correlation_heatmap(df)

        generate_markdown_report(report_data)

        # 保存最终数据集
        selected_features = [
            'birth_month',
            'reg_address_encoded',
            'main_profession_encoded',
            'school_encoded',
            'major_code_encoded',
            'major_name_encoded',
            'sex_enc',
            'nation_enc',
            'marriage_enc',
            'edu_level_enc',
            'politic_enc',
            'religion_enc',
            'type_enc',
            'military_status_enc',
            'is_disability_enc',
            'is_elder_enc',
            'is_teen_enc',
            'label'
        ]

        output_path = "../processed_data/final_processed_data.csv"
        df[selected_features].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n最终数据集已保存至：{output_path}")

    except Exception as e:
        print(f"\n执行过程中发生严重错误：{str(e)}")
        print("建议检查：")
        print("1. 输入文件路径是否正确")
        print("2. 数据是否包含非数值型列")
        print("3. Python依赖库版本是否兼容")