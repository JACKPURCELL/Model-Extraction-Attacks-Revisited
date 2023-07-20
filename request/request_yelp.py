
import csv
import json
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#我这里.py文件和数据放在同一个路径下了，如果不在同一个路径下，自己可以修改，注意路径要用//

json_file_path='/data/jc/data/sentiment/YELP/yelp_academic_dataset_review.json'
# csv_file_path='/data/jc/data/sentiment/YELP/yelp_academic_dataset_review.csv'
path = '/data/jc/data/sentiment/YELP'
#打开business.json文件,取出第一行列名
with open(json_file_path,'r',encoding='utf-8') as fin:
    for line in fin:
        line_contents = json.loads(line)
        headers=line_contents.keys()
        break
    print(headers)

count = 0
#将json读成字典,其键值写入business.csv的列名,再将json文件中的values逐行写入business.csv文件

with open(json_file_path, 'r', encoding='utf-8') as fin:
    for line in tqdm(fin):
        count+=1
        line_contents = json.loads(line)
        text_content = line_contents['text'].replace('\n', ' ')
        star_content = line_contents['stars']
        prefix = 'pos' if star_content > 3.0 else 'neg'
        file_name = 'yelp_' + str(count)
        with open(os.path.join(path, prefix, file_name+'.txt'), mode='w') as f:
            f.write(text_content)
            
#  # 删除state','postal_code','is_open','attributes'列,并保存
#  # 可以根据需要选择，这里是针对review文件的一些列。
# df_bus=pd.read_csv(csv_file_path)
# df_reduced=df_bus.drop(['user_id','business_id','useful','funny','cool', 'date'], axis = 1)

# # df_reduced=df_bus.drop(['compliment_hot','compliment_more','compliment_profile'],axis=1)
# df_cleaned=df_reduced.dropna()
# df_cleaned.to_csv(csv_file_path,index=False)
# df_bus=pd.read_csv(csv_file_path)

# df_bus.to_csv(csv_file_path,index=False)
# :qa