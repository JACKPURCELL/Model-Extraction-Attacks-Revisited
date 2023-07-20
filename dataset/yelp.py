"""
Script containing various utilities related to data processing and cleaning. Includes tokenization,
text cleaning, feature extractor (token type IDs & attention masks) for BERT, and YELPDataset.
"""

import json
import logging
import shutil
import torch
from torch.utils.data import Dataset

import os
import pickle
import re
import numpy as np
from tqdm import trange
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Setup stopwords list & word (noun, adjective, and verb) lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()




class YELP(Dataset):
    """
    YELP Dataset for easily iterating over and performing common operations.

    @param (str) input_directory: path of directory where the desired data exists
    @param (pytorch_transformers.BertTokenizer) tokenizer: tokenizer with pre-figured mappings
    @param (bool) apply_cleaning: whether or not to perform common cleaning operations on texts;
           note that enabling only makes sense if language of the task is English
    @param (int) max_tokenization_length: maximum number of positional embeddings, or the sequence
           length of an example that will be fed to BERT model (default: 512)log_dir
    @param (str) truncation_method: method that will be applied in case the text exceeds
           @max_tokenization_length; currently implemented methods include 'head-only', 'tail-only',
           and 'head+tail' (default: 'head-only')
    @param (float) split_head_density: weight on head when splitting between head and tail, only
           applicable if @truncation_method='head+tail' (default: 0.5)
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the data tensors

    """
    def __init__(self, input_directory,tokenizer,retokenize,api,max_length,log_dir):
        self.input_directory = input_directory
        self.positive_path = os.path.join(input_directory, 'pos')
        self.positive_files = [f for f in os.listdir(self.positive_path)
                               if os.path.isfile(os.path.join(self.positive_path, f))]
        self.num_positive_examples = len(self.positive_files)
        self.positive_label = 0
        self.negative_path = os.path.join(input_directory, 'neg')
        self.negative_files = [f for f in os.listdir(self.negative_path)
                               if os.path.isfile(os.path.join(self.negative_path, f))]
        self.num_negative_examples = len(self.negative_files)
        self.negative_label = 1
        self.log_dir=log_dir
        self.tokenizer = tokenizer
        self.retokenize = retokenize
        self.api = api
        self.max_length = max_length

        self.norm_par=None

        self.info_lb = torch.ones(50010,dtype=torch.long)*(-1)
        self.info_conf = torch.zeros(50010)

            
        # Pre-tokenize & encode examples
        self.pre_tokenize_and_encode_examples()

    def pre_tokenize_and_encode_examples(self):
        """
        Function to tokenize & encode examples and save the tokenized versions to a separate folder.
        This way, we won't have to perform the same tokenization and encoding ops every epoch.
        """
        if self.retokenize or not os.path.exists(os.path.join(self.positive_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir)):
            if self.retokenize:
                try:
                    shutil.rmtree(os.path.join(self.positive_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir))
                except:
                    print("no path")   
            os.mkdir(os.path.join(self.positive_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir))

            # Clean & tokenize positive reviews
            for i in trange(len(self.positive_files), desc='Tokenizing & Encoding Positive Reviews',
                            leave=True):
                file = self.positive_files[i]
                with open(os.path.join(self.positive_path, file), mode='r', encoding='utf8') as f:
                    example = f.read()

                example = re.sub(r'<br />', '', example)
                example = example.lstrip().rstrip()
                example = re.sub(' +', ' ', example)
                example = self.tokenizer(example,return_tensors='pt',padding="max_length",max_length=self.max_length,truncation=True)

                with open(os.path.join(self.positive_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir, file), mode='wb') as f:
                    pickle.dump(obj=example, file=f)
        else:
            logging.warning('Tokenized positive reviews directory already exists!')

        if self.retokenize or not os.path.exists(os.path.join(self.negative_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir)):
            if self.retokenize:
                try:
                    shutil.rmtree(os.path.join(self.negative_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir))
                except:
                    print("no path")   
            os.mkdir(os.path.join(self.negative_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir))

            # Clean & tokenize negative reviews
            for i in trange(len(self.negative_files), desc='Tokenizing & Encoding Negative Reviews',
                            leave=True):
                file = self.negative_files[i]
                with open(os.path.join(self.negative_path, file), mode='r', encoding='utf8') as f:
                    example = f.read()

                example = re.sub(r'<br />', '', example)
                example = example.lstrip().rstrip()
                example = re.sub(' +', ' ', example)
                example = self.tokenizer(example,return_tensors='pt',padding="max_length",max_length=self.max_length,truncation=True)


                with open(os.path.join(self.negative_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir, file), mode='wb') as f:
                    pickle.dump(obj=example, file=f)
        else:
            logging.warning('Tokenized negative reviews directory already exists!')

    def __len__(self):
        return len(self.positive_files) + len(self.negative_files)

    def __getitem__(self, index):
        if index < self.num_positive_examples:
            file = self.positive_files[index]
            label = torch.tensor(data=self.positive_label, dtype=torch.long)
            with open(os.path.join(self.positive_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir, file), mode='rb') as f:
                example = pickle.load(file=f)
            match self.api:
                case 'amazon':
                    with open(os.path.join(self.input_directory, 'amazon_api', file), mode='rb') as p:
                        api_result = json.load(p)
                case 'microsoft':
                    with open(os.path.join(self.input_directory, 'mic_api_2022-11-01', file), mode='rb') as p:
                        api_result = json.load(p)
                case _:
                    pass
        elif index >= self.num_positive_examples:
            file = self.negative_files[index-self.num_positive_examples]
            label = torch.tensor(data=self.negative_label, dtype=torch.long)
            with open(os.path.join(self.negative_path, 'tokenized_and_encoded' + self.tokenizer.name_or_path+self.log_dir, file), mode='rb') as f:
                example = pickle.load(file=f)
            match self.api:
                case 'amazon':
                    with open(os.path.join(self.input_directory, 'amazon_api', file), mode='rb') as p:
                        api_result = json.load(p)
                case 'microsoft':
                    with open(os.path.join(self.input_directory, 'mic_api_2022-11-01', file), mode='rb') as p:
                        api_result = json.load(p)
                case _:
                    pass
        else:
            raise ValueError('Out of range index while accessing dataset')

        match self.api:
            case 'amazon':
                soft_label = torch.ones(4)
                soft_label[0] = api_result['SentimentScore']['Positive']
                soft_label[1] = api_result['SentimentScore']['Negative']
                soft_label[2] = api_result['SentimentScore']['Mixed']
                soft_label[3] = api_result['SentimentScore']['Neutral']
                hapi_label = torch.argmax(soft_label,dim=-1)
                # if soft_label[0] >= soft_label[1]:
                #     hapi_label = torch.tensor(0)
                # else:
                #     hapi_label = torch.tensor(1)  
            case 'microsoft':
                soft_label = torch.ones(3)
                soft_label[0] = api_result[0]['positive']
                soft_label[1] = api_result[0]['negative']
                soft_label[2] = api_result[0]['neutral']
                hapi_label = torch.argmax(soft_label,dim=-1)
                
                # if soft_label[0] >= soft_label[1]:
                #     hapi_label = torch.tensor(0)
                # else:
                #     hapi_label = torch.tensor(1)
            
        if 'roberta' in self.tokenizer.name_or_path or 't5' in self.tokenizer.name_or_path:
            return example.input_ids[0], example.attention_mask[0], label, soft_label, hapi_label
        return example.input_ids[0], example.token_type_ids[0], example.attention_mask[0], label, soft_label, hapi_label
