import pickle
import re
import boto3
import json
import os

from tqdm import trange
# This example requires environment variables named "LANGUAGE_KEY" and "LANGUAGE_ENDPOINT"
language_key = '30a6cca823254b4f9251a37934c4cf32'
language_endpoint = 'https://hapitwtw.cognitiveservices.azure.com/'

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Authenticate the client using your key and endpoint 
def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=language_endpoint, 
            credential=ta_credential,model_version='2021-10-01')
    return text_analytics_client

client = authenticate_client()
# (positive=0.54, neutral=0.06, negative=0.4) result[0].sentiment result[0].confidence_scores.positive SentimentConfidenceScores(positive=0.54, neutral=0.06, negative=0.4)
# Sentiment Analysis and opinion mining	2021-10-01, 2022-06-01,2022-10-01,2021-10-01*
# Example method for detecting sentiment and opinions in text 
def sentiment_analysis(documents):

    # documents = [
    #     "The food and service were unacceptable. The concierge was nice, however."
    # ]


    result = client.analyze_sentiment(documents)
    data = [{
            'positive' : result[0].confidence_scores.positive,
            'neutral' : result[0].confidence_scores.neutral,
            'negative' : result[0].confidence_scores.negative,
            'sentiment' : result[0].sentiment   
            }]

    data = json.dumps(data)
    
    return data,result
   
          
mic_api_path = 'mic_api_2021-10-01'
mic_api_ori_path = 'mic_api_ori_2021-10-01'


input_directory = '/data/jc/data/sentiment/IMDB/aclImdb/train'
positive_path = os.path.join(input_directory, 'pos')
positive_files = [f for f in os.listdir(positive_path)
                        if os.path.isfile(os.path.join(positive_path, f))]
num_positive_examples = len(positive_files)
positive_label = 0
negative_path = os.path.join(input_directory, 'neg')
negative_files = [f for f in os.listdir(negative_path)
                        if os.path.isfile(os.path.join(negative_path, f))]
num_negative_examples = len(negative_files)
        
try:        
    os.mkdir(os.path.join(positive_path, mic_api_path))
    os.mkdir(os.path.join(positive_path, mic_api_ori_path))
except:
    pass
for i in trange(len(positive_files), desc='Tokenizing & Encoding Positive Reviews',
                leave=True):
    file = positive_files[i]
    if not os.path.exists(os.path.join(positive_path, mic_api_path, file)):
        with open(os.path.join(positive_path, file), mode='r', encoding='utf8') as f:
            example = f.read()

        example = re.sub(r'<br />', '', example)
        example = example.lstrip().rstrip()
        example = re.sub(' +', ' ', example)
        if len(example) > 4950:
            example=example[:4950]

        data,result = sentiment_analysis([example])

        with open(os.path.join(positive_path, mic_api_path, file), mode='w') as f:
            f.write(data)
        with open(os.path.join(positive_path, mic_api_ori_path, file), mode='wb') as f:
            pickle.dump(result, f)
try:
    os.mkdir(os.path.join(negative_path, mic_api_ori_path))
    os.mkdir(os.path.join(negative_path, mic_api_path))
except:
    pass
# for i in trange(len(negative_files), desc='Tokenizing & Encoding negative_files Reviews',
#                 leave=True):
#     file = negative_files[i]
#     if not os.path.exists(os.path.join(negative_path, mic_api_path, file)):
#         with open(os.path.join(negative_path, file), mode='r', encoding='utf8') as f:
#             example = f.read()

#         example = re.sub(r'<br />', '', example)
#         example = example.lstrip().rstrip()
#         example = re.sub(' +', ' ', example)
#         if len(example) > 4950:
#             example=example[:4950]
#         data,result = sentiment_analysis([example])


#         with open(os.path.join(negative_path, mic_api_path, file), mode='w') as f:
#             f.write(data)
#         with open(os.path.join(negative_path, mic_api_ori_path, file), mode='wb') as f:
#             pickle.dump(result, f)
            
            