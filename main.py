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

    
def split_dataset(dataset: Dataset | Subset,
                  length: int = None, percent: float = None,
                  shuffle: bool = True
                  ) -> tuple[Subset, Subset]:
    r"""Split a dataset into two subsets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        length (int): The length of the first subset.
            This argument cannot be used together with :attr:`percent`.
            If ``None``, use :attr:`percent` to calculate length instead.
            Defaults to ``None``.
        percent (float): The split ratio for the first subset.
            This argument cannot be used together with :attr:`length`.
            ``length = percent * len(dataset)``.
            Defaults to ``None``.
        shuffle (bool): Whether to shuffle the dataset.
            Defaults to ``True``.
        seed (bool): The random seed to split dataset
            using :any:`numpy.random.shuffle`.
            Defaults to ``None``.

    Returns:
        (torch.utils.data.Subset, torch.utils.data.Subset):
            The two splitted subsets.

    :Example:
        >>> import torch
        >>> from trojanzoo.utils.data import TensorListDataset, split_dataset
        >>>
        >>> data = torch.ones(11, 3, 32, 32)
        >>> targets = list(range(11))
        >>> dataset = TensorListDataset(data, targets)
        >>> set1, set2 = split_dataset(dataset, length=3)
        >>> len(set1), len(set2)
        (3, 8)
        >>> set3, set4 = split_dataset(dataset, percent=0.5)
        >>> len(set3), len(set4)
        (5, 6)

    Note:
        This is the implementation of :meth:`trojanzoo.datasets.Dataset.split_dataset`.
        The difference is that this method will NOT set :attr:`seed`
        as ``env['data_seed']`` when it is ``None``.
    """
    assert (length is None) != (percent is None)  # XOR check
    length = length if length is not None else int(len(dataset) * percent)
    #TODO: if batch_size != 64
    if length % 64 != 0:
        length += 64 - (length % 64)
        if length > len(dataset):
            length -= 64
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    if isinstance(dataset, Subset):
        idx = np.array(dataset.indices)
        indices = idx[indices]
        dataset = dataset.dataset
    subset1 = Subset(dataset, indices[:length])
    subset2 = Subset(dataset, indices[length:])
    return subset1, subset2


parser = argparse.ArgumentParser(description='Process some integers.')

# os.environ["CUDA_VISIBLE_DEVICES"] =0,1
# TRUNCATION_METHOD = 'head-only'


parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--lr_warmup_percent', type=float, default=0.1)
parser.add_argument('--custom_lr', type=float, default=1e-3)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--bert_weight_decay', type=float, default=0.01)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hapi_info', type=str, default='sa/imdb/amazon_sa/22-05-23')
parser.add_argument('--hapi_data_dir', type=str, default='/home/jkl6486/HAPI')
parser.add_argument('--dataset_path', type=str, default='/data/jc/data/sentiment/IMDB/')
parser.add_argument('--model', type=str, default='xlnet-base-cased')
parser.add_argument('--dataset', type=str, default='imdb')
parser.add_argument('--log_dir', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--mixmatch', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--op_parameters', type=str, default='partial')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--validate_interval', type=int, default=1)
#TODO: fix save
parser.add_argument('--save', action='store_true')
parser.add_argument('--label_train', action='store_true')
parser.add_argument('--retokenize', action='store_true')
parser.add_argument('--split_unlabel_percent', type=float, default=0.0)
parser.add_argument('--split_label_percent', type=float, default=1.0)

parser.add_argument('--api', type=str)

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

# Initialize train & test datasets
if args.dataset == 'imdb':  
    train_dataset = IMDB(input_directory=os.path.join(args.dataset_path,"aclImdb/test"),tokenizer=model.tokenizer,hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,retokenize=args.retokenize,api=args.api,max_length=args.max_length)
    test_dataset = IMDB(input_directory=os.path.join(args.dataset_path,"aclImdb/train"),tokenizer=model.tokenizer,hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,retokenize=args.retokenize,api=args.api,max_length=args.max_length)
    task = 'sentiment'
elif args.dataset == 'rafdb':
    train_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api)
    test_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api)
    task = 'emotion'

if args.split_unlabel_percent != 0.0:

    
    _temp_dataset, _ = split_dataset(
        train_dataset,
        percent=args.split_label_percent+args.split_unlabel_percent)
    
    
    _label_dataset, _temp_unlabel_dataset = split_dataset(
        _temp_dataset,
        percent=args.split_label_percent/(args.split_label_percent+args.split_unlabel_percent))
    
    _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'mixmatch'),
                                _temp_unlabel_dataset.indices)

    train_loader = DataLoader(dataset=_label_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,drop_last=True)
    _unlabel_dataloader = DataLoader(dataset=_unlabel_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = itertools.cycle(_unlabel_dataloader)
    
    
elif args.split_label_percent != 1.0:
    _label_dataset, _ = split_dataset(
        train_dataset,
        percent=args.split_label_percent)
    train_loader = DataLoader(dataset=_label_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = None

else:
    train_loader = DataLoader(dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = None
    
            
            
# Acquire iterators through data loaders


test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,drop_last=True)



    

    
optimizer, lr_scheduler = model.define_optimizer(
        parameters=args.op_parameters,
        OptimType=args.optimizer,
        lr=args.lr,custom_lr=args.custom_lr,weight_decay=args.bert_weight_decay,
        lr_scheduler=args.lr_scheduler,
        epochs=args.epochs, 
        lr_warmup_percent=args.lr_warmup_percent, 
        betas=args.betas,
        eps=args.eps)


 
if args.log_dir:
    log_dir = 'runs/'+args.hapi_info.replace('/','_')+"_ep"+str(args.epochs)+"_num_classes_"+str(args.num_classes)+"_lr"+str(args.lr)+"_bs"+str(args.batch_size)+"_"+args.optimizer+"_"+args.op_parameters+"_"+args.model
    if args.label_train:
        log_dir += "_labeltrain"
    if args.api:
        log_dir += args.api
    if args.mixmatch:
        log_dir += "_mixmatch"

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
    # for key, value in vars(args).items():
    #     f.write('%s:%s\n'%(key, str(value)))
    #     print(key, value) 
# parser = ArgumentParser()
# args = parser.parse_args()
# with open('commandline_args.txt', 'r') as f:
#     args.__dict__ = json.load(f)                  
distillation(module=model,num_classes=args.num_classes,
             validate_interval=args.validate_interval,
             epochs=args.epochs,optimizer=optimizer,
             lr_scheduler=lr_scheduler, 
             log_dir=log_dir,
             grad_clip=args.grad_clip,
             loader_train=train_loader,loader_valid=test_loader,unlabel_iterator=unlabel_iterator,
             mixmatch=args.mixmatch,
             save=args.save,label_train=args.label_train,
             api=args.api,task=task)
