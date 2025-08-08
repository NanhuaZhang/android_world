import pandas as pd
import os

from test import getKeys

csv_path = 'output_all.csv'

target_ids = getKeys()
# 读取原始CSV
df = pd.read_csv(csv_path,encoding='utf-8')

# 筛选出id在数组中的行
filtered_df = df[df['id'].isin(target_ids)]

# 保存为新CSV文件
filtered_df.to_csv('filtered_ids.csv', index=False)