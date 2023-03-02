import pickle
import re
import boto3
import json
import os


from tqdm import trange

old_mic_api_path = 'mic_api_2021-10-01'

mic_api_path = 'mic_api_2022-11-01'

input_directory = '/data/jc/data/sentiment/IMDB/aclImdb/train'
positive_path = os.path.join(input_directory, 'pos')
positive_files = [f for f in os.listdir(positive_path)
                        if os.path.isfile(os.path.join(positive_path, f))]
num_positive_examples = len(positive_files)

negative_path = os.path.join(input_directory, 'neg')
negative_files = [f for f in os.listdir(negative_path)
                        if os.path.isfile(os.path.join(negative_path, f))]
num_negative_examples = len(negative_files)
        

diff = 0
same = 0
pos = 0
for i in trange(len(positive_files), desc='Tokenizing & Encoding Positive Reviews',
                leave=True):
    file = positive_files[i]
    if os.path.exists(os.path.join(positive_path, mic_api_path, file)):
        with open(os.path.join(positive_path, mic_api_path, file), mode='r') as f:
            data = f.read()
            datanew = json.loads(data)
            # if datanew[0]['sentiment'] != 'mixed':
            #     pos+=1
        with open(os.path.join(positive_path, old_mic_api_path, file), mode='r') as f:
            data = f.read()
            dataold = json.loads(data)
        if sorted(datanew[0].items()) == sorted(dataold[0].items()):
            same +=1
        else:
            diff +=1
print(same, diff)
# print(pos)
