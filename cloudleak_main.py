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
# from pytorch_transformers import AdamW
from dataset.expw import EXPW  # Adam's optimization w/ fixed weight decay
from dataset.ferplus import FERPLUS

from dataset.imdb import IMDB
from dataset.rafdb import RAFDB
from dataset.kdef import KDEF
from dataset.cifar import CIFAR10
from utils.cloudleak import distillation
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
parser.add_argument('--lr_warmup_epoch', type=int, default=0)
parser.add_argument('--custom_lr', type=float, default=1e-3)
parser.add_argument('--betas', type=tuple)
# parser.add_argument('--betas', type=tuple, default=(0.9, 0.99))
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--grad_clip', type=float)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hapi_info', type=str, default='sa/imdb/amazon_sa/22-05-23')
parser.add_argument('--hapi_data_dir', type=str, default='/home/jkl6486/HAPI')
parser.add_argument('--dataset_path', type=str, default='/data/jc/data/sentiment/IMDB_hapi/')
parser.add_argument('--model', type=str, default='xlnet-base-cased')
parser.add_argument('--dataset', type=str, default='imdb')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='Lion')
parser.add_argument('--mixmatch', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--op_parameters', type=str, default='partial')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--validate_interval', type=int, default=1)
parser.add_argument('--save', action='store_true')
parser.add_argument('--label_train', action='store_true')
parser.add_argument('--hapi_label_train', action='store_true')
parser.add_argument('--retokenize', action='store_true')
parser.add_argument('--split_unlabel_percent', type=float, default=0.0)
parser.add_argument('--split_label_percent', type=float, default=1.0)

parser.add_argument('--unlabel_batch', type=int, default=-1)
parser.add_argument('--label_batch', type=int, default=-1)
#adv:clean 64:0 48:16=3 32:32=1 16:48=1/3
parser.add_argument('--adv_percent', type=float, default=-1)

parser.add_argument('--pgd_percent', type=float)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--adaptive', choices=['entropy', 'kcenter','random','cloudleak'])
parser.add_argument('--n_samples', type=int)
parser.add_argument('--sample_times', type=int)

parser.add_argument('--api', type=str)
parser.add_argument('--lr_scheduler_freq', type=str,default='epoch')
parser.add_argument('--adv_train', choices=[None, 'pgd', 'free', 'cw'],
                           help='adversarial training (default: None)')
parser.add_argument('--adv_valid', choices=[None, 'pgd', 'free', 'cw'])

parser.add_argument('--encoder_train', action='store_true')
parser.add_argument('--encoder_path',  type=str)
parser.add_argument('--teacher_path',  type=str)
parser.add_argument('--encoder_attack', action='store_true')

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
if args.optimizer == 'Lion':
    args.betas = (0.9, 0.99)
else:
    args.betas = (0.9, 0.999)

if args.optimizer == 'Lion' and args.dataset == 'imdb':
    args.betas = (0.95, 0.98)
elif args.dataset == 'imdb':
    args.betas = (0.9, 0.99)
# PRETRAINED_MODEL_NAME = 'bert-base-cased'
# NUM_PRETRAINED_BERT_LAYERS = 4
# MAX_TOKENIZATION_LENGTH = 512


if args.encoder_train:
    transform = 'raw'
else:
    transform = 'Normal'
if args.dataset == 'rafdb':
    train_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    test_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    task = 'emotion'
elif args.dataset == 'kdef':
    train_dataset = KDEF(input_directory=os.path.join('/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    test_dataset = KDEF(input_directory=os.path.join('/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    task = 'emotion'
elif args.dataset == 'ferplus':
    train_dataset = FERPLUS(input_directory=os.path.join('/data/jc/data/image/ferplus_hapi',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    test_dataset = FERPLUS(input_directory=os.path.join('/data/jc/data/image/ferplus_hapi',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    task = 'emotion'
elif args.dataset == 'cifar10':
    train_dataset = CIFAR10(mode='train',transform=transform)
    test_dataset = CIFAR10(mode='valid',transform=transform)
    task = 'cifar10'
elif args.dataset == 'expw':
    train_dataset = EXPW(input_directory=os.path.join('/data/jc/data/image/EXPW_224',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    test_dataset = EXPW(input_directory=os.path.join('/data/jc/data/image/EXPW_224',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    task = 'emotion'
    
print(args.model.split('-')[0])

if args.log_dir is None:
    log_dir = 'runs/'+"ep"+str(args.epochs)+"_nc_"+str(args.num_classes)+"_lr"+str(args.lr)+"_bs"+str(args.batch_size)+"_"+args.optimizer+"_"+args.op_parameters+"_"+args.model+"_per_"+str(args.split_label_percent)
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
    if args.adv_train:
        log_dir += "_advtrain"
    if args.pgd_percent:
        log_dir += str(args.pgd_percent)
    if args.lr_scheduler:
        log_dir += "_lrsche"
else:
    log_dir = 'submit3/'+args.log_dir
    
if 'resnet' in args.model:
    model = getattr(models,'resnet')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'xlnet' in args.model:
    model = getattr(models,'xlnet')(model_name=args.model,num_classes=args.num_classes)
elif 'vit' in args.model:
    model = getattr(models,'vit')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'roberta' in args.model:
    model = getattr(models,'roberta')(model_name=args.model,num_classes=args.num_classes)
elif 'vgg' in args.model:
    model = getattr(models,'vgg')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'autoencoder' in args.model:
    model = getattr(models,'autoencoder')(norm_par=train_dataset.norm_par)
if args.api == 'cifar10': 
    tea_model =  getattr(models,'resnet')(norm_par=train_dataset.norm_par,model_name='resnet50_cifar10',num_classes=args.num_classes)    
    tea_model.load_state_dict(args.teacher_path)
else:
    tea_model = None
if parallel:
    model.model = nn.DataParallel(model.model).cuda()    
# model.load_state_dict(torch.load('/home/jkl6486/hermes/runs/fer_rafdb_facepp_fer_22-05-23_ep50_num_classes_7_lr0.0003_bs64_Lion_full_resnet50_percent_1.0_labeltrainfacepp_lr_schedulerdropout/model.pth'))
# Initialize train & test datasets
if args.dataset == 'imdb':
    if 'hapi' not in args.api:
        path =  '/data/jc/data/sentiment/IMDB_api/'
        train_dataset = IMDB(input_directory=os.path.join(path,"aclImdb/train"),tokenizer=model.tokenizer,hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,retokenize=args.retokenize,api=args.api,max_length=args.max_length,log_dir=args.log_dir)
        test_dataset = IMDB(input_directory=os.path.join(path,"aclImdb/test"),tokenizer=model.tokenizer,hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,retokenize=args.retokenize,api=args.api,max_length=args.max_length,log_dir=args.log_dir)
    else:
        path = '/data/jc/data/sentiment/IMDB_hapi/'
        print(path)
        train_dataset = IMDB(input_directory=os.path.join(path,"aclImdb/test"),tokenizer=model.tokenizer,hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,retokenize=args.retokenize,api=args.api,max_length=args.max_length,log_dir=args.log_dir)
        test_dataset = IMDB(input_directory=os.path.join(path,"aclImdb/train"),tokenizer=model.tokenizer,hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,retokenize=args.retokenize,api=args.api,max_length=args.max_length,log_dir=args.log_dir)
    task = 'sentiment'

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
if args.unlabel_batch != -1:#mixmatchseed
    _temp_dataset, _ = split_dataset(
        train_dataset,
        length=(args.label_batch+args.unlabel_batch)*args.batch_size)
    
    
    _label_dataset, _temp_unlabel_dataset = split_dataset(
        _temp_dataset,length=args.label_batch*args.batch_size)

    if args.dataset == 'rafdb':
        _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'mixmatch'),
                                _temp_unlabel_dataset.indices)
    elif args.dataset == 'kdef':
        _unlabel_dataset = Subset(KDEF(input_directory=os.path.join('/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'mixmatch'),
                                _temp_unlabel_dataset.indices)
    elif args.dataset == 'ferplus':
        _unlabel_dataset = Subset(FERPLUS(input_directory=os.path.join('/data/jc/data/image/ferplus_hapi',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = transform_type),
                                _temp_unlabel_dataset.indices)
    elif args.dataset == 'cifar10':
        _unlabel_dataset = Subset(CIFAR10(mode='train',transform = 'mixmatch'),
                                _temp_unlabel_dataset.indices)
    elif args.dataset == 'expw':
        _unlabel_dataset = Subset(EXPW(input_directory=os.path.join('/data/jc/data/image/EXPW_224',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'mixmatch'),
                                _temp_unlabel_dataset.indices)
    else:
        raise NotImplementedError
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
    
    
elif args.label_batch != -1:#adaptive or part data
    
    _label_dataset, _unlabel_dataset = split_dataset(
        train_dataset,length=args.label_batch*args.batch_size)

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
    # /home/jkl6486/hermes/exp_adv_encoder/encoder_train_rafdb/model.pth
if args.encoder_attack:
    AE = getattr(models,'autoencoder')(norm_par=train_dataset.norm_par)            
    AE.load_state_dict(torch.load(args.encoder_path))   
    for p in AE.parameters():
        p.requires_grad = False
else:
    AE = None
# Acquire iterators through data loaders
if args.adaptive == 'kcenter' or args.adaptive == 'random':
    train_loader = None

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
        lr_warmup_epoch=args.lr_warmup_epoch,
        betas=args.betas,
        eps=args.eps,total_iters=total_iters)


 

    



try:
    os.mkdir(log_dir)
except:
    print( "Directory %s already exists" % log_dir)


with open(os.path.join(log_dir,'args.txt'), mode='w') as f:
    json.dump(args.__dict__, f, indent=2)
    
    
with open(os.path.join(log_dir, "train.csv"), "w") as f:
    f.write("epoch,hapi_loss,hapi_acc1\n")

with open(os.path.join(log_dir, "valid.csv"), "w") as f:
    f.write("epoch,gt_loss,gt_acc1,hapi_loss,hapi_acc1,adv_fidelity,adv_fidelity_hard\n")
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
             save=args.save,label_train=args.label_train,
             hapi_label_train=args.hapi_label_train, lr_scheduler_freq=args.lr_scheduler_freq,
             api=args.api,task=task,unlabel_dataset_indices=unlabel_dataset_indices,
             hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,
             batch_size=args.batch_size,num_workers=args.num_workers,
             n_samples = args.n_samples,adaptive=args.adaptive,get_sampler_fn=get_sampler,
             balance=args.balance,sample_times=args.sample_times,tea_model=tea_model,AE=AE,encoder_attack=args.encoder_attack,pgd_percent=args.pgd_percent,
             encoder_train=args.encoder_train,adv_percent=args.adv_percent,train_dataset=train_dataset,workers=args.num_workers)
