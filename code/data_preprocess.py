import os
import pandas as pd
from utils import load_data, load_data_pre

data = load_data("../raw_data/附件1.xls")
data_pre = load_data_pre("../raw_data/附件1.xls")
data_pre = data_pre.iloc[:, :-1]

key_columns = ['age', 'sex', 'nation', 'edu_level', 'c_aac009']
data = data.dropna(subset=key_columns, how='any')
data_pre = data_pre.dropna(subset=key_columns, how='any')

data = data.drop_duplicates(subset=['people_id'])
data_pre = data_pre.drop_duplicates(subset=['people_id'])

os.makedirs("../processed_data", exist_ok=True)

data.to_csv(os.path.join("../processed_data/data.csv"), index=False)
data_pre.to_csv(os.path.join("../processed_data/data_pre.csv"), index=False)

print("Successfully Save to data.csv")
print("Successfully Save to data_pre.csv")