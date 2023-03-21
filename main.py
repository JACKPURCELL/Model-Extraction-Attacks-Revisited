"""
Script for training, testing, and saving finetuned, binary classification models based on pretrained
BERT parameters, for the IMDB dataset.
"""

import itertools

import json
import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# !pip install pytorch_transformers
from pytorch_transformers import AdamW  # Adam's optimization w/ fixed weight decay

from dataset.imdb import IMDB
from dataset.rafdb import RAFDB
from utils.model import distillation,train
from torch.utils.data import Dataset,Subset


from models.xlnet import XLNet
import models
import argparse
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import split_dataset



parser = argparse.ArgumentParser(description='Process some integers.')

# os.environ["CUDA_VISIBLE_DEVICES"] =0,1
# TRUNCATION_METHOD = 'head-only'


parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float)
parser.add_argument('--lr_warmup_percent', type=float, default=0.0)
parser.add_argument('--custom_lr', type=float, default=1e-3)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--grad_clip', type=float)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hapi_info', type=str, default='sa/imdb/amazon_sa/22-05-23')
parser.add_argument('--hapi_data_dir', type=str, default='/home/jkl6486/HAPI')
parser.add_argument('--dataset_path', type=str, default='/data/jc/data/sentiment/IMDB_hapi/')
parser.add_argument('--model', type=str, default='xlnet-base-cased')
parser.add_argument('--dataset', type=str, default='imdb')
parser.add_argument('--log_dir', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--mixmatch', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--op_parameters', type=str, default='partial')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--validate_interval', type=int, default=1)
parser.add_argument('--save', action='store_true')
parser.add_argument('--label_train', action='store_true')
parser.add_argument('--retokenize', action='store_true')
parser.add_argument('--split_unlabel_percent', type=float, default=0.0)
parser.add_argument('--split_label_percent', type=float, default=1.0)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--n_samples', type=int)
parser.add_argument('--sample_times', type=int)

parser.add_argument('--api', type=str)
parser.add_argument('--lr_scheduler_freq', type=str,default='epoch')
parser.add_argument('--adv_train', choices=[None, 'pgd', 'free', 'trades'],
                           help='adversarial training (default: None)')
parser.add_argument('--adv_valid', action='store_true')

parser.add_argument('--adv_train_iter', type=int, default=7)

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

if 'resnet' in args.model:
    model = getattr(models,'resnet')(parallel=parallel,model_name=args.model,num_classes=args.num_classes)
elif 'xlnet' in args.model:
    model = getattr(models,'xlnet')(parallel=parallel,model_name=args.model,num_classes=args.num_classes)
elif 'vit' in args.model:
    model = getattr(models,'vit')(parallel=parallel,model_name=args.model,num_classes=args.num_classes)
elif 'roberta' in args.model:
    model = getattr(models,'roberta')(parallel=parallel,model_name=args.model,num_classes=args.num_classes)
elif 'vgg' in args.model:
    model = getattr(models,'vgg')(parallel=parallel,model_name=args.model,num_classes=args.num_classes)
# Initialize train & test datasets
if args.dataset == 'imdb':  
    train_dataset = IMDB(input_directory=os.path.join(args.dataset_path,"aclImdb/test"),tokenizer=model.tokenizer,hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,retokenize=args.retokenize,api=args.api,max_length=args.max_length)
    test_dataset = IMDB(input_directory=os.path.join(args.dataset_path,"aclImdb/train"),tokenizer=model.tokenizer,hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,retokenize=args.retokenize,api=args.api,max_length=args.max_length)
    task = 'sentiment'
elif args.dataset == 'rafdb':
    train_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api)
    test_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api)
    task = 'emotion'
    
def get_sampler(train_dataset):
    from collections import Counter
    from torch.utils.data.sampler import WeightedRandomSampler    
    class_counts = []
    result = Counter(train_dataset.targets) 
    for i in range(len(result)):
        class_counts.append(result[i])

    num_samples = sum(class_counts)
    labels = train_dataset.targets #corresponding labels of samples

    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    return WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

unlabel_dataset_indices =None
if args.split_unlabel_percent != 0.0:#mixmatch
    _temp_dataset, _ = split_dataset(
        train_dataset,
        percent=args.split_label_percent+args.split_unlabel_percent)
    
    
    _label_dataset, _temp_unlabel_dataset = split_dataset(
        _temp_dataset,
        percent=args.split_label_percent/(args.split_label_percent+args.split_unlabel_percent))
    
    _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'mixmatch'),
                                _temp_unlabel_dataset.indices)
    if args.balance:
        sampler=get_sampler(_label_dataset)
        shuffle = False
    else:
        sampler=None
        shuffle = True
    train_loader = DataLoader(dataset=_label_dataset,
                    batch_size=args.batch_size,
                    shuffle=shuffle,sampler=sampler,
                    num_workers=args.num_workers,drop_last=True)
    _unlabel_dataloader = DataLoader(dataset=_unlabel_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = itertools.cycle(_unlabel_dataloader)
    
    
elif args.split_label_percent != 1.0:#adaptive or part data
    
    _label_dataset, _unlabel_dataset = split_dataset(
        train_dataset,
        percent=args.split_label_percent)
    unlabel_dataset_indices=_unlabel_dataset.indices
    if args.balance:
        sampler=get_sampler(_label_dataset)
        shuffle = False
    else:
        sampler=None
        shuffle = True
        

    train_loader = DataLoader(dataset=_label_dataset,
                    batch_size=args.batch_size,
                    shuffle=shuffle,sampler=sampler,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = None

else:
    if args.balance:
        sampler=get_sampler(train_dataset)
        shuffle = False
    else:
        sampler=None
        shuffle = True
    train_loader = DataLoader(dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=shuffle,sampler=sampler,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = None
    
            
            
# Acquire iterators through data loaders


test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,drop_last=True)



    
total_iters = len(train_dataset)/args.batch_size*args.epochs
    
optimizer, lr_scheduler = model.define_optimizer(
        parameters=args.op_parameters,
        OptimType=args.optimizer,
        lr=args.lr,custom_lr=args.custom_lr,weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        epochs=args.epochs, 
        lr_warmup_percent=args.lr_warmup_percent, 
        betas=args.betas,
        eps=args.eps,total_iters=total_iters)


 
if args.log_dir:
    log_dir = 'runs/'+args.hapi_info.replace('/','_')+"_ep"+str(args.epochs)+"_num_classes_"+str(args.num_classes)+"_lr"+str(args.lr)+"_bs"+str(args.batch_size)+"_"+args.optimizer+"_"+args.op_parameters+"_"+args.model+"_percent_"+str(args.split_label_percent)
    if args.label_train:
        log_dir += "_labeltrain"
    if args.api:
        log_dir += args.api
    if args.mixmatch:
        log_dir += "_mixmatch"
    if args.balance:
        log_dir += "_balance"
    if args.adaptive:
        log_dir += "adaptive"
else:
    log_dir = 'runs/debug'
    
if args.lr_scheduler:
    log_dir += "_lr_scheduler"

log_dir += "dropout"

try:
    os.mkdir(log_dir)
except:
    print( "Directory %s already exists" % log_dir)


with open(os.path.join(log_dir,'args.txt'), mode='w') as f:
    json.dump(args.__dict__, f, indent=2)
    # for key, value in vars(args).items():
    #     f.write('%s:%s\n'%(key, str(value)))
    #     print(key, value) 
# parser = ArgumentParser()
# args = parser.parse_args()
# with open('commandline_args.txt', 'r') as f:
#     args.__dict__ = json.load(f)                  
distillation(module=model,pgd_set = test_dataset,adv_train=args.adv_train,num_classes=args.num_classes,
             validate_interval=args.validate_interval,
             epochs=args.epochs,optimizer=optimizer,
             lr_scheduler=lr_scheduler, 
             adv_train_iter=args.adv_train_iter,adv_valid=args.adv_valid,adv_valid_iter=args.adv_valid,
             log_dir=log_dir,
             grad_clip=args.grad_clip,
             loader_train=train_loader,loader_valid=test_loader,unlabel_iterator=unlabel_iterator,
             mixmatch=args.mixmatch,
             save=args.save,label_train=args.label_train,lr_scheduler_freq=args.lr_scheduler_freq,
             api=args.api,task=task,unlabel_dataset_indices=_unlabel_dataset.indices if args.adaptive else None,
             hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,
             batch_size=args.batch_size,num_workers=args.num_workers,
             n_samples = args.n_samples,adaptive=args.adaptive,get_sampler_fn=get_sampler,balance=args.balance,sample_times=args.sample_times)
