import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib


sns.set_theme(style="whitegrid")

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


data_path = "data/cleaned_data.csv"
plots_dir = "results/plots"
os.makedirs(plots_dir, exist_ok=True)

data = pd.read_csv(data_path, skiprows=[1])

data['性别'] = data['sex'].map({1: '男', 2: '女'}).fillna('未知')

edu_mapping = {
    '10': '研究生',
    '20': '本科',
    '30': '专科',
    '90': '文盲'
}
data['学历'] = data['edu_level'].astype(str).map(edu_mapping).fillna('未知')

#就业状态计算
from datetime import datetime
check_date = pd.to_datetime("2023-12-31")

def determine_employment_status(row, check_date):
    employment_date = pd.to_datetime(row['b_acc031'], errors='coerce')
    unemployment_date = pd.to_datetime(row['c_ajc090'], errors='coerce')
    cancel_unemployment_date = pd.to_datetime(row['c_acc028'], errors='coerce')

    if pd.notnull(employment_date) and pd.notnull(unemployment_date):
        if unemployment_date < employment_date:
            return '在职'

    if pd.notnull(employment_date) and employment_date <= check_date:
        if pd.isnull(unemployment_date) or unemployment_date > check_date:
            if pd.isnull(cancel_unemployment_date) or cancel_unemployment_date > check_date:
                return '在职'
            elif cancel_unemployment_date <= check_date:
                return '非失业'
        elif unemployment_date <= check_date:
            return '失业'
    return '未知'

data['就业状态'] = data.apply(lambda row: determine_employment_status(row, check_date), axis=1)
data['年龄'] = pd.to_numeric(data['age'], errors='coerce')
data['年龄'] = data['年龄'].fillna(data['年龄'].mean())

# ========= 图1：性别 vs 就业率 =========
data['在职状态'] = data['就业状态'].apply(lambda x: 1 if x == '在职' else 0)
plt.figure(figsize=(6, 4))
sns.barplot(x='性别', y='在职状态', data=data, estimator='mean', errorbar=None, palette='pastel')
for i, rate in enumerate(data.groupby('性别')['在职状态'].mean()):
    plt.text(i, rate + 0.02, f"{rate:.2f}", ha='center', fontsize=10)
plt.ylim(0, 1)
plt.title("性别对就业状态的影响", fontsize=14)
plt.ylabel("就业率", fontsize=12)
plt.xlabel("性别", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{plots_dir}/性别就业率.png", dpi=300)
plt.close()

# ========= 图2：年龄分组 vs 就业率 =========
bins = [18, 25, 35, 45, 55, 65]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65']
data['年龄分组'] = pd.cut(data['年龄'], bins=bins, labels=labels)
age_group_rate = data.groupby('年龄分组', observed=False)['就业状态'].apply(
    lambda x: (x == '在职').mean()).reset_index()

# 计算合理上下限范围
y_min = max(0, age_group_rate['就业状态'].min() - 0.05)
y_max = min(1, age_group_rate['就业状态'].max() + 0.05)

plt.figure(figsize=(8, 4))
sns.lineplot(x='年龄分组', y='就业状态', data=age_group_rate, marker='o', linewidth=2, color='#66b3ff')
for x, y in zip(age_group_rate['年龄分组'], age_group_rate['就业状态']):
    plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=10)
plt.ylim(y_min, y_max)
plt.title("不同年龄段的就业率变化", fontsize=14)
plt.ylabel("就业率", fontsize=12)
plt.xlabel("年龄分组", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{plots_dir}/年龄分组就业率.png", dpi=300)
plt.close()

# ========= 图3：学历 vs 就业率 =========
edu_rate = data.groupby('学历', observed=False)['就业状态'].apply(
    lambda x: (x == '在职').mean()).sort_values().reset_index()
plt.figure(figsize=(8, 4))
sns.barplot(x='就业状态', y='学历', data=edu_rate, palette='Set2')
for i, (edu, rate) in enumerate(zip(edu_rate['学历'], edu_rate['就业状态'])):
    plt.text(rate + 0.02, i, f"{rate:.2f}", va='center', fontsize=10)
plt.xlim(0, 1)
plt.title("学历对就业率的影响", fontsize=14)
plt.xlabel("就业率", fontsize=12)
plt.ylabel("学历", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{plots_dir}/学历就业率.png", dpi=300)
plt.close()

# ========= 图4：就业状态分布（饼图） =========
employment_distribution = data['就业状态'].value_counts()
colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999']
plt.figure(figsize=(6, 6))
employment_distribution.plot.pie(
    autopct=lambda p: f"{p:.1f}%" if p > 0 else '',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 11}
)
plt.title("就业状态分布", fontsize=14)
plt.ylabel('')
plt.tight_layout()
plt.savefig(f"{plots_dir}/就业状态分布.png", dpi=300)
plt.close()


# ========= 图5,6：失业时间段分析 =========
# 将 '失业时间' 列转换为日期格式
data['失业时间'] = pd.to_datetime(data['c_ajc090'], errors='coerce')

# 按年分组统计失业人数
data['失业年份'] = data['失业时间'].dt.year

# 过滤掉没有失业时间的数据
unemployed_data = data[data['失业时间'].notna()]

# 绘制失业人数按年分布图
plt.figure(figsize=(8, 5))
sns.countplot(x='失业年份', data=unemployed_data, palette='Set2')
plt.title("各年份失业人数分布", fontsize=14)
plt.xlabel("年份", fontsize=12)
plt.ylabel("人数", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{plots_dir}/失业人数按年分布.png", dpi=300)
plt.close()

# 按月分组：
unemployed_data['失业月份'] = unemployed_data['失业时间'].dt.month
plt.figure(figsize=(8, 5))
sns.countplot(x='失业月份', data=unemployed_data, palette='Set2')
plt.title("各月份失业人数分布", fontsize=14)
plt.xlabel("月份", fontsize=12)
plt.ylabel("人数", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{plots_dir}/失业人数按月分布.png", dpi=300)
plt.close()

data.to_csv("results/cleaned_for_analysis.csv", index=False)

print("✅ 四张分析图已保存到目录: results/plots")
print("✅ 已保存清洗后的分析数据：results/cleaned_for_analysis.csv")
