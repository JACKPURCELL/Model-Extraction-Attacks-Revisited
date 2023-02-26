import pickle
import re
import boto3
import json
import os

from tqdm import trange
# comprehend = boto3.client(service_name='comprehend', region_name='us-east-2')



# input_directory = '/data/jc/data/sentiment/IMDB/aclImdb/test'
# positive_path = os.path.join(input_directory, 'pos')
# positive_files = [f for f in os.listdir(positive_path)
#                         if os.path.isfile(os.path.join(positive_path, f))]
# num_positive_examples = len(positive_files)
# positive_label = 0
# negative_path = os.path.join(input_directory, 'neg')
# negative_files = [f for f in os.listdir(negative_path)
#                         if os.path.isfile(os.path.join(negative_path, f))]
# num_negative_examples = len(negative_files)
        
        
# os.mkdir(os.path.join(positive_path, 'amazon_api'))
# for i in trange(len(positive_files), desc='Tokenizing & Encoding Positive Reviews',
#                 leave=True):
#     file = positive_files[i]
#     if not os.path.exists(os.path.join(positive_path, 'amazon_api', file)):
#         with open(os.path.join(positive_path, file), mode='r', encoding='utf8') as f:
#             example = f.read()

#         example = re.sub(r'<br />', '', example)
#         example = example.lstrip().rstrip()
#         example = re.sub(' +', ' ', example)
#         if len(example) > 4950:
#             example=example[:4950]
#         amz_return = json.dumps(comprehend.detect_sentiment(Text=example, LanguageCode='en'), sort_keys=True, indent=4)

#         with open(os.path.join(positive_path, 'amazon_api', file), mode='w') as f:
#             f.write(amz_return)


# os.mkdir(os.path.join(negative_path, 'amazon_api'))
# for i in trange(len(negative_files), desc='Tokenizing & Encoding negative_files Reviews',
#                 leave=True):
#     file = negative_files[i]
#     with open(os.path.join(negative_path, file), mode='r', encoding='utf8') as f:
#         example = f.read()

#     example = re.sub(r'<br />', '', example)
#     example = example.lstrip().rstrip()
#     example = re.sub(' +', ' ', example)
#     if len(example) > 4950:
#         example=example[:4950]
#     amz_return = json.dumps(comprehend.detect_sentiment(Text=example, LanguageCode='en'), sort_keys=True, indent=4)

#     with open(os.path.join(negative_path, 'amazon_api', file), mode='w') as f:
#         f.write(amz_return)
        
        
with open(os.path.join('/data/jc/data/sentiment/IMDB/aclImdb/test/neg/amazon_api/21_1.txt'), mode='rb') as p:
    amazon = json.load(p)
    
print(amazon)