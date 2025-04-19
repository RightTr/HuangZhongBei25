import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import os

# 选择合适的字体路径
font_path = "C:\\Windows\\Fonts\\simsun.ttc"  # 或者使用 SimSun

if os.path.exists(font_path):
    my_font = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = my_font.get_name()  # 设置全局字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方框
    print(f"✅ 已设置中文字体为: {my_font.get_name()}")
else:
    print("❌ 字体未找到，请检查路径！")

# ===== 数据读取与处理 =====
df = pd.read_csv('../processed_data/df_result.csv')

# 性别映射
sex_mapping = {0: '未知', 1: '男', 2: '女', 9: '未说明'}
df['sex_性别'] = df['sex_性别'].map(sex_mapping).fillna('未知')

# 教育程度映射
edu_mapping = {
    10: '研究生教育', 11: '博士研究生毕业', 12: '博士研究生结业', 13: '博士研究生肄业',
    14: '硕士研究生毕业', 15: '硕士研究生结业', 16: '硕士研究生肄业', 17: '研究生班毕业',
    18: '研究生班结业', 19: '研究生班肄业', 20: '大学本科教育', 21: '大学本科毕业',
    22: '大学本科结业', 23: '大学本科肄业', 28: '大学普通班毕业', 30: '大学专科教育',
    31: '大学专科毕业', 32: '大学专科结业', 33: '大学专科肄业', 40: '中等职业教育',
    41: '中等专科毕业', 42: '中等专科结业', 43: '中等专科肄业', 44: '职业高中毕业',
    45: '职业高中结业', 46: '职业高中肄业', 47: '技工学校毕业', 48: '技工学校结业',
    49: '技工学校肄业', 50: '高中以下', 60: '普通高级中学教育', 61: '普通高中毕业',
    62: '普通高中结业', 63: '普通高中肄业', 70: '初级中学教育', 71: '初中毕业',
    73: '初中肄业', 80: '小学教育', 81: '小学毕业', 83: '小学肄业', 90: '文盲或半文盲',
    91: '中等师范学校（幼儿师范学校）毕业', 92: '中等师范学校（幼儿师范学校）结业',
    93: '中等师范学校（幼儿师范学校）肄业', 99: '其他'
}
df['edu_level_教育程度'] = df['edu_level_教育程度'].map(edu_mapping).fillna('未知')

# 毕业年过滤
min_grad_year = df['graduate_year'].min()
df_filtered = df[df['graduate_year'] > min_grad_year]

# 输出目录
output_dir = '../figure'
os.makedirs(output_dir, exist_ok=True)

# 就业率数据准备
employment_rate_by_grad_year = df_filtered.groupby('graduate_year')['label'].mean()
employment_rate_by_age = df.groupby('age_年龄')['label'].mean()

# ===== 图表绘制 =====

# 1. 性别与就业状态
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sex_性别', hue='label', hue_order=[0, 1])
plt.title('性别与就业状态的关系', fontsize=16)
plt.xlabel('性别', fontsize=14)
plt.ylabel('人数', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='就业状态', labels=['无业', '已就业'], fontsize=12, title_fontsize=13, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sex_vs_employment_status.png'))
plt.close()

# 2. 教育程度与就业状态
plt.figure(figsize=(12, 10))
sns.countplot(data=df, y='edu_level_教育程度', hue='label', hue_order=[0, 1])
plt.title('教育程度与就业状态的关系', fontsize=20)
plt.xlabel('人数', fontsize=14)
plt.ylabel('教育程度', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='就业状态', labels=['无业', '已就业'], fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'edu_level_vs_employment_status.png'))
plt.close()

# 3. 年龄与就业率折线图
plt.figure(figsize=(10, 6))
sns.lineplot(x=employment_rate_by_age.index, y=employment_rate_by_age.values, marker='o')
max_idx = employment_rate_by_age.idxmax()
plt.annotate(f'峰值：{employment_rate_by_age.max():.2f}', xy=(max_idx, employment_rate_by_age.max()),
             xytext=(max_idx + 1, employment_rate_by_age.max() - 0.1),
             arrowprops=dict(arrowstyle='->', color='red'))
plt.title('年龄与就业状态的关系（就业率趋势）', fontsize=16)
plt.xlabel('年龄', fontsize=14)
plt.ylabel('就业率', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'age_vs_employment_rate.png'))
plt.close()

# 4. 毕业年份与就业率
plt.figure(figsize=(10, 6))
sns.lineplot(data=employment_rate_by_grad_year, marker='o')
plt.scatter(employment_rate_by_grad_year.index, employment_rate_by_grad_year.values,
            color='red', label='数据点')
plt.title('毕业年份与就业状态的关系', fontsize=16)
plt.xlabel('毕业年份', fontsize=14)
plt.ylabel('就业率', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'graduate_year_vs_employment_status_with_markers.png'))
plt.close()

# 5. 年龄与就业状态箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='label', y='age_年龄', hue='label', palette='coolwarm')
plt.title('年龄与就业状态的箱线图', fontsize=16)
plt.xlabel('就业状态', fontsize=14)
plt.ylabel('年龄', fontsize=14)
plt.xticks([0, 1], ['无业', '已就业'], fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'age_vs_employment_boxplot.png'))
plt.close()

# 6. 性别与就业状态饼图
sex_label_counts = df.groupby(['sex_性别', 'label']).size().unstack().fillna(0)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
labels = ['无业', '已就业']
for i, col in enumerate(sex_label_counts.columns):
    axes[i].pie(sex_label_counts[col], labels=sex_label_counts.index,
                autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
    axes[i].set_title(f'就业状态：{labels[col]}', fontsize=14)
plt.suptitle('性别与就业状态的分布（饼图）', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sex_vs_employment_pie.png'))
plt.close()

# 7. 多变量散点图（PairPlot）
sns.pairplot(df, hue='label',
             vars=['age_年龄', 'years_since_grad', 'reg_address_encoded', 'main_profession_encoded'],
             hue_order=[0, 1], palette='Set1', diag_kind='kde',
             plot_kws={'alpha': 0.6, 's': 50})
plt.suptitle('变量与就业状态的关系（PairPlot）', fontsize=16, y=1.02)
plt.savefig(os.path.join(output_dir, 'features_vs_employment_pairplot.png'))
plt.close()

print("✅ 所有图表已保存至 'figure' 文件夹")
