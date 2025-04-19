
# 数据特征分析报告
**生成时间**：2025-04-19 04:01:05

## 1. 目标变量分布

|   就业状态 |   数量 |
|-----------:|-------:|
|          1 |   3983 |
|          0 |    393 |

## 2. 特征共线性分析（VIF）

| 特征名称                                        |     VIF值 | 共线性评价   |
|:------------------------------------------------|----------:|:-------------|
| <span style='color:red'>graduate_year</span>    | inf       | nan          |
| <span style='color:red'>years_since_grad</span> | inf       | nan          |
| <span style='color:red'>birth_year</span>       | 507.423   | 严重共线性   |
| <span style='color:red'>age_年龄</span>         | 502.127   | 严重共线性   |
| <span style='color:red'>const</span>            | 173.576   | 严重共线性   |
| birth_month                                     |   1.5265  | 无共线性     |
| major_code_encoded                              |   1.34212 | 无共线性     |
| marriage_婚姻状态_enc                           |   1.32836 | 无共线性     |
| major_name_encoded                              |   1.32638 | 无共线性     |
| is_disability_是否残疾人_enc                    |   1.13802 | 无共线性     |
| is_living_alone_是否独居_enc                    |   1.13573 | 无共线性     |
| edu_level_教育程度_enc                          |   1.06562 | 无共线性     |
| is_elder_是否老年人_enc                         |   1.05483 | 无共线性     |
| sex_性别_enc                                    |   1.04797 | 无共线性     |
| military_status_兵役状态_enc                    |   1.04453 | 无共线性     |
| religion_宗教信仰_enc                           |   1.03965 | 无共线性     |
| nation_民族_enc                                 |   1.03086 | 无共线性     |
| main_profession_encoded                         |   1.02772 | 无共线性     |
| reg_address_encoded                             |   1.02265 | 无共线性     |
| school_encoded                                  |   1.02017 | 无共线性     |
| politic_政治面貌_enc                            |   1.01846 | 无共线性     |
| type_人口类型_enc                               |   1.00841 | 无共线性     |
| is_teen_是否青少年_enc                          | nan       | nan          |
| change_type_变动类型_enc                        | nan       | nan          |

**共线性说明**：
- 严重共线性（VIF>50）：红色标记特征
- 显著共线性（VIF>10）：橙色标记特征
- 轻度共线性（VIF>5）：黄色标记特征

## 3. 特征相关性热力图
![相关系数矩阵](./figure/correlation_matrix.jpg)

## 4. 分析建议

### 数据质量问题
- 检测到 5 个高共线性特征（VIF>10）

### 特征工程建议
1. **共线性处理**：
   - 删除特征：`<span style='color:red'>graduate_year</span>, <span style='color:red'>years_since_grad</span>, <span style='color:red'>birth_year</span>, <span style='color:red'>age_年龄</span>, <span style='color:red'>const</span>`
   - 使用PCA降维（保留85%方差）
2. **特征优化**：
   - 离散化处理：`birth_month` 转换为季度特征
   - 嵌入编码：`main_profession_encoded` 使用FastText编码

### 建模建议
- 样本重采样（SMOTE过采样）
- 使用LightGBM内置类别特征处理
- 添加特征交互项：`school_encoded × major_code_encoded`
        