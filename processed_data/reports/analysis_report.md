
# 数据特征分析报告
**生成时间**：2025-04-19 10:23:08

## 1. 目标变量分布

|   就业状态 |   数量 |
|-----------:|-------:|
|          1 |   4253 |
|          0 |    420 |

## 2. 特征共线性分析（VIF）

| 特征名称                                        |     VIF值 | 共线性评价   |
|:------------------------------------------------|----------:|:-------------|
| <span style='color:red'>graduate_year</span>    | inf       | nan          |
| <span style='color:red'>years_since_grad</span> | inf       | nan          |
| <span style='color:red'>birth_year</span>       | 510.261   | 严重共线性   |
| <span style='color:red'>age</span>              | 486.956   | 严重共线性   |
| birth_month                                     |   1.52326 | 无共线性     |
| const                                           |   1.36164 | 无共线性     |
| marriage_enc                                    |   1.32629 | 无共线性     |
| is_disability_enc                               |   1.12845 | 无共线性     |
| is_living_alone_enc                             |   1.12677 | 无共线性     |
| edu_level_enc                                   |   1.06552 | 无共线性     |
| is_elder_enc                                    |   1.05749 | 无共线性     |
| military_status_enc                             |   1.04208 | 无共线性     |
| sex_enc                                         |   1.04198 | 无共线性     |
| religion_enc                                    |   1.03529 | 无共线性     |
| reg_address_encoded                             |   1.02038 | 无共线性     |
| school_encoded                                  |   1.01761 | 无共线性     |
| politic_enc                                     |   1.01644 | 无共线性     |
| nation_enc                                      |   1.01582 | 无共线性     |
| type_enc                                        |   1.00664 | 无共线性     |
| is_teen_enc                                     | nan       | nan          |
| change_type_enc                                 | nan       | nan          |

**共线性说明**：
- 严重共线性（VIF>50）：红色标记特征
- 显著共线性（VIF>10）：橙色标记特征
- 轻度共线性（VIF>5）：黄色标记特征

## 3. 特征相关性热力图
![相关系数矩阵](./figure/correlation_matrix.jpg)

## 4. 分析建议

### 数据质量问题
- 检测到 4 个高共线性特征（VIF>10）

### 特征工程建议
1. **共线性处理**：
   - 删除特征：`<span style='color:red'>graduate_year</span>, <span style='color:red'>years_since_grad</span>, <span style='color:red'>birth_year</span>, <span style='color:red'>age</span>`
   - 使用PCA降维（保留85%方差）
2. **特征优化**：
   - 离散化处理：`birth_month` 转换为季度特征
   - 嵌入编码：`main_profession_encoded` 使用FastText编码

### 建模建议
- 样本重采样（SMOTE过采样）
- 使用LightGBM内置类别特征处理
- 添加特征交互项：`school_encoded × major_code_encoded`
        