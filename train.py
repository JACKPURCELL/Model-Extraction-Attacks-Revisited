"""
Script for training, testing, and saving finetuned, binary classification models based on pretrained
BERT parameters, for the IMDB dataset.
"""


import json
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# !pip install pytorch_transformers
from pytorch_transformers import AdamW  # Adam's optimization w/ fixed weight decay

from dataset.imdb import IMDB
from utils.model import distillation,train
from models.xlnet import XLNet
import models
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

# os.environ["CUDA_VISIBLE_DEVICES"] =0,1
# TRUNCATION_METHOD = 'head-only'


parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--bert_lr', type=float, default=5e-5)
parser.add_argument('--lr_warmup_percent', type=float, default=0.1)
# parser.add_argument('--custom_lr', type=float, default=1e-3)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--bert_weight_decay', type=float, default=0.01)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hapi_info', type=str, default='sa/imdb/amazon_sa/22-05-23')
parser.add_argument('--dataset_path', type=str, default='/data/jc/data/sentiment/IMDB_new/')
parser.add_argument('--model', type=str, default='xlnet-base-cased')
parser.add_argument('--log_dir', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--mixmatch', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--op_parameters', type=str, default='full')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--validate_interval', type=int, default=1)
#TODO: fix save
parser.add_argument('--save', action='store_true')
parser.add_argument('--label_train', action='store_true')
parser.add_argument('--retokenize', action='store_true')
args = parser.parse_args()

device = torch.cuda.device_count()
parallel = True if device > 1 else False
print("DEVICE FOUND: %s" % device)

# Set args.seeds for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Define hyperparameters


# PRETRAINED_MODEL_NAME = 'bert-base-cased'
# NUM_PRETRAINED_BERT_LAYERS = 4
# MAX_TOKENIZATION_LENGTH = 512


print(args.model.split('-')[0])
model = getattr(models,args.model.split('-')[0])(parallel=parallel,model_name=args.model)

# Initialize train & test datasets
train_dataset = IMDB(input_directory=os.path.join(args.dataset_path,"aclImdb/test"),tokenizer=model.tokenizer,hapi_info=args.hapi_info,retokenize=args.retokenize)

test_dataset = IMDB(input_directory=os.path.join(args.dataset_path,"aclImdb/train"),tokenizer=model.tokenizer,hapi_info=args.hapi_info,retokenize=args.retokenize)

# Acquire iterators through data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers,drop_last=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,drop_last=True)



    
    
optimizer, lr_scheduler = model.define_optimizer(
        parameters=args.op_parameters,
        OptimType=args.optimizer,
        lr=args.bert_lr,weight_decay=args.bert_weight_decay,
        lr_scheduler=args.lr_scheduler,
        epochs=args.epochs, 
        lr_warmup_percent=args.lr_warmup_percent, 
        betas=args.betas,
        eps=args.eps)


 
if args.log_dir:
    log_dir = 'runs/'+ 'train'+args.hapi_info.replace('/','_')+"_ep"+str(args.epochs)+"_lr"+str(args.bert_lr)+"_bs"+str(args.batch_size)+"_"+args.optimizer
    if args.label_train:
        log_dir += "_labeltrain"
else:
    log_dir = 'runs/debug'

if args.lr_scheduler:
    log_dir += "_lr_scheduler"

try:
    os.mkdir(log_dir)
except:
    print( "Directory %s already exists" % log_dir)


with open(os.path.join(log_dir,'args.txt'), mode='w') as f:
    json.dump(args.__dict__, f, indent=2)
                
train(module=model,num_classes=args.num_classes,
             validate_interval=args.validate_interval,
             epochs=args.epochs,optimizer=optimizer,
             lr_scheduler=lr_scheduler, 
             log_dir=log_dir,
             grad_clip=args.grad_clip,
             loader_train=train_loader,loader_valid=test_loader,
             mixmatch=args.mixmatch,
             save=args.save,label_train=args.label_train)
