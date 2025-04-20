import pandas as pd
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import jieba
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


df_raw = pd.read_csv("../processed_data/data.csv", encoding='utf-8')
df = df_raw.iloc[1:].reset_index(drop=True)
df.replace('\\N', pd.NA, inplace=True)

today = pd.to_datetime(datetime.today().date())

date_cols = [
    'b_acc031', # 就业时间
    'b_aae031', # 合同终止日期
    'c_acc028' # 失业注销时间
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df['label'] = 0
df.loc[df['c_acc0m3'] == '就业', 'label'] = 1
df.loc[df['b_acc031'].notna() & (df['b_aae031'].isna() | (df['b_aae031'] > today)),'label'] = 1
df.loc[df['c_acc028'].notna(), 'label'] = 1

# 教育程度映射
df_resume = df[["教育程度", "所学专业名称", "毕业学校"]]
def map_education_level(code):
    education_map = {
        10: "研究生教育", 11: "博士研究生毕业", 12: "博士研究生结业", 13: "博士研究生肄业",
        14: "硕士研究生毕业", 15: "硕士研究生结业", 16: "硕士研究生肄业", 17: "研究生班毕业",
        18: "研究生班结业", 19: "研究生班肄业", 20: "大学本科教育", 21: "大学本科毕业",
        22: "大学本科结业", 23: "大学本科肄业", 28: "大学普通班毕业", 30: "大学专科教育",
        31: "大学专科毕业", 32: "大学专科结业", 33: "大学专科肄业", 40: "中等职业教育",
        41: "中等专科毕业", 42: "中等专科结业", 43: "中等专科肄业", 44: "职业高中毕业",
        45: "职业高中结业", 46: "职业高中肄业", 47: "技工学校毕业", 48: "技工学校结业",
        49: "技工学校肄业", 50: "高中以下", 60: "普通高级中学教育", 61: "普通高中毕业",
        62: "普通高中结业", 63: "普通高中肄业", 70: "初级中学教育", 71: "初中毕业",
        73: "初中肄业", 80: "小学教育", 81: "小学毕业", 83: "小学肄业", 90: "文盲或半文盲",
        91: "中等师范学校（幼儿师范学校）毕业", 92: "中等师范学校（幼儿师范学校）结业",
        93: "中等师范学校（幼儿师范学校）肄业", 99: "其他"
    }
    try:
        return education_map.get(int(code), "未知")
    except:
        return "未知"

df_resume.loc[:, '教育程度_文字'] = df_resume['教育程度'].apply(map_education_level)
df_resume.loc[:, '简历文本'] = df_resume.fillna('').astype(str).agg(' '.join, axis=1)
df_resume = df_resume[["教育程度", "所学专业名称", "毕业学校", "简历文本"]]

# 增加岗位数据
data = [
    {'岗位名称': '数据分析师', '岗位职责': '负责数据清洗、建模、分析', '任职要求': '掌握Python，熟练使用Excel、SQL'},
    {'岗位名称': '会计', '岗位职责': '负责公司财务报表', '任职要求': '具备会计相关证书，熟悉报税流程'},
    {'岗位名称': '客服专员', '岗位职责': '负责用户接待与问题反馈', '任职要求': '良好的沟通能力，耐心细致'},
    {'岗位名称': '产品经理', '岗位职责': '负责产品规划与开发，协调各部门工作', '任职要求': '具备一定的项目管理经验，沟通能力强'},
    {'岗位名称': 'UI设计师', '岗位职责': '设计软件界面，提升用户体验', '任职要求': '熟练掌握设计工具，具有较强的美术功底'},
    {'岗位名称': '前端开发工程师', '岗位职责': '负责网站和应用的前端开发', '任职要求': '精通HTML、CSS、JavaScript，了解前端框架'},
    {'岗位名称': '后端开发工程师', '岗位职责': '开发与维护后台服务和数据库', '任职要求': '熟悉Java、Python等开发语言，了解常见数据库'},
    {'岗位名称': '销售经理', '岗位职责': '负责销售团队的管理与客户关系维护', '任职要求': '具有较强的销售能力，良好的团队管理经验'},
    {'岗位名称': 'HR专员', '岗位职责': '负责员工招聘、培训与绩效管理', '任职要求': '熟悉人力资源管理流程，有一定的沟通技巧'},
    {'岗位名称': '数据科学家', '岗位职责': '分析数据，提供决策支持，构建预测模型', '任职要求': '精通数据分析与机器学习，有相关工作经验'},
    {'岗位名称': '市场分析师', '岗位职责': '对市场进行调研，分析竞争对手及市场趋势', '任职要求': '具备一定的市场分析能力，良好的数据处理能力'},
    {'岗位名称': '客户经理', '岗位职责': '维护客户关系，推进销售目标的达成', '任职要求': '具备良好的客户沟通技巧与销售经验'},
    {'岗位名称': '网络工程师', '岗位职责': '负责公司网络设备的配置与维护', '任职要求': '了解网络协议，具备网络设备管理经验'},
    {'岗位名称': '系统管理员', '岗位职责': '维护公司IT系统，确保系统稳定运行', '任职要求': '精通Linux或Windows系统管理，具备一定的网络安全知识'},
    {'岗位名称': '法务专员', '岗位职责': '处理公司法务事务，提供法律咨询', '任职要求': '法学相关专业，有律师资格证者优先'},
    {'岗位名称': '金融分析师', '岗位职责': '对金融市场进行分析，预测未来趋势', '任职要求': '具备金融分析的基本知识，熟练使用金融分析工具'}
]

df_jobs = pd.DataFrame(data)
df_jobs['岗位文本'] = df_jobs[['岗位名称', '岗位职责', '任职要求']].agg(' '.join, axis=1)

# 使用 TF-IDF 进行文本向量化
vectorizer = TfidfVectorizer(max_features=5000)  # 限制特征数目，防止维度过高

# 向量化岗位和简历文本
resume_tfidf = vectorizer.fit_transform(df_resume['简历文本'].tolist())
job_tfidf = vectorizer.transform(df_jobs['岗位文本'].tolist())

# 计算相似度并推荐岗位
similarity_matrix = cosine_similarity(resume_tfidf, job_tfidf)  # 计算每个简历对每个岗位的相似度

# 输出推荐结果
print("\n===== 岗位推荐结果 =====\n")
for i, resume in df_resume.iterrows():
    sim_scores = similarity_matrix[i]
    best_match_idx = sim_scores.argmax()
    print(f"简历 {i+1} 推荐岗位：{df_jobs.iloc[best_match_idx]['岗位名称']}（相似度：{sim_scores[best_match_idx]:.2f}）")