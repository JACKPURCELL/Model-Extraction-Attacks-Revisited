import pickle
import re
import shutil
import boto3
import json
import os
from random import sample
import random
from tqdm import trange
comprehend = boto3.client(service_name='comprehend', region_name='us-east-2')

random.seed(1688)
path = '/data/jc/data/sentiment/YELP'
# os.mkdir(os.path.join(path, 'amazon_api'))
# os.mkdir(os.path.join(path, 'train'))
# os.mkdir(os.path.join(path, 'train', 'pos'))
# os.mkdir(os.path.join(path, 'train', 'neg'))
# os.mkdir(os.path.join(path, 'valid'))
# os.mkdir(os.path.join(path, 'valid', 'pos'))
# os.mkdir(os.path.join(path, 'valid', 'neg'))

# positive_path = os.path.join(path, 'pos')
# positive_files = [f for f in os.listdir(positive_path)
#                         if os.path.isfile(os.path.join(positive_path, f))]


negative_path = os.path.join(path, 'neg')
negative_files = [f for f in os.listdir(negative_path)
                        if os.path.isfile(os.path.join(negative_path, f))]
        
# positive_files = sample(positive_files, 25000)       
# for i in trange(len(positive_files), desc='Tokenizing & Encoding Positive Reviews',
#                 leave=True):
#     file = positive_files[i]
#     if i < 12500:
#         shutil.copy(os.path.join(positive_path, file), os.path.join(path, 'train', 'pos'))
#     else:
#         shutil.copy(os.path.join(positive_path, file), os.path.join(path, 'valid', 'pos'))
#     if not os.path.exists(os.path.join(path, 'amazon_api', file)):
#         with open(os.path.join(positive_path, file), mode='r', encoding='utf8') as f:
#             example = f.read()

#         example = re.sub(r'<br />', '', example)
#         example = example.lstrip().rstrip()
#         example = re.sub(' +', ' ', example)
#         if len(example) > 4950:
#             example=example[:4950]
#         amz_return = json.dumps(comprehend.detect_sentiment(Text=example, LanguageCode='en'), sort_keys=True, indent=4)

#         with open(os.path.join(path,'amazon_api', file), mode='w') as f:
#             f.write(amz_return)

path = '/data/jc/data/sentiment/YELP'

negative_files = sample(negative_files, 25000)       
for i in trange(len(negative_files), desc='Tokenizing & Encoding negative Reviews',
                leave=True):
    file = negative_files[i]
    if i < 12500:
        shutil.copy(os.path.join(negative_path, file), os.path.join(path, 'train', 'neg'))
    else:
        shutil.copy(os.path.join(negative_path, file), os.path.join(path, 'valid', 'neg'))
    if not os.path.exists(os.path.join(path, 'amazon_api', file)):
        with open(os.path.join(negative_path, file), mode='r', encoding='utf8') as f:
            example = f.read()

        example = re.sub(r'<br />', '', example)
        example = example.lstrip().rstrip()
        example = re.sub(' +', ' ', example)
        if len(example) > 4950:
            example=example[:4950]
        amz_return = json.dumps(comprehend.detect_sentiment(Text=example, LanguageCode='en'), sort_keys=True, indent=4)

        with open(os.path.join(path,'amazon_api', file), mode='w') as f:
            f.write(amz_return)
