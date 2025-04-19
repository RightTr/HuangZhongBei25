import os
import pandas as pd
from utils import load_data

data = load_data("../raw_data/附件1.xls")

key_columns = ['age', 'sex', 'nation', 'edu_level', 'c_aac009']
data = data.dropna(subset=key_columns, how='any')

data = data.drop_duplicates(subset=['people_id'])

os.makedirs("../processed_data", exist_ok=True)

print(data)

data.to_csv(os.path.join("../processed_data/data.csv"), index=False)
print("Successfully Save to data.csv")