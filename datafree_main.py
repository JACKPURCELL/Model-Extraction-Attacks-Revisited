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
from torch.optim.optimizer import Optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# !pip install pytorch_transformers
# from pytorch_transformers import AdamW
from dataset.expw import EXPW  # Adam's optimization w/ fixed weight decay

from dataset.imdb import IMDB
from dataset.yelp import YELP
from dataset.rafdb import RAFDB
from dataset.ferplus import FERPLUS
from dataset.kdef import KDEF
from dataset.cifar import CIFAR10
from utils.datafree import distillation
from torch.utils.data import Dataset,Subset


from models.xlnet import XLNet
import models
import argparse
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import split_dataset



parser = argparse.ArgumentParser(description='Process some integers.')

# os.environ["CUDA_VISIBLE_DEVICES"] =0,1
# TRUNCATION_METHOD = 'head-only'

#----------
parser.add_argument('--query_budget', type=float, default=20, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
parser.add_argument('--epoch_itrs', type=int, default=50)  
parser.add_argument('--g_iter', type=int, default=1, help = "Number of generator iterations per epoch_iter")
parser.add_argument('--d_iter', type=int, default=5, help = "Number of discriminator iterations per epoch_iter")

# parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
parser.add_argument('--nz', type=int, default=256, help = "Size of random noise input to generator")


parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'kl'],)

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')

# parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')



parser.add_argument('--approx_grad', type=int, default=1, help = 'Always set to 1')
parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
parser.add_argument('--grad_epsilon', type=float, default=1e-3) 


parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')


# Eigenvalues computation parameters
parser.add_argument('--no_logits', type=int, default=1)
parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])

parser.add_argument('--rec_grad_norm', type=int, default=1)



8
#--------
parser.add_argument('--batch_size', type=int, default=64)
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
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--op_parameters', type=str, default='full')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--validate_interval', type=int, default=1)
parser.add_argument('--save', action='store_true')
parser.add_argument('--label_train', action='store_true')
parser.add_argument('--hapi_label_train', action='store_true')


parser.add_argument('--api', type=str)
parser.add_argument('--lr_scheduler_freq', type=str,default='epoch')

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

if args.optimizer == 'Lion' and (args.dataset == 'imdb' or args.dataset == 'yelp'):
    args.betas = (0.95, 0.98)
elif args.dataset == 'imdb' or args.dataset == 'yelp':
    args.betas = (0.9, 0.99)
# PRETRAINED_MODEL_NAME = 'bert-base-cased'
# NUM_PRETRAINED_BERT_LAYERS = 4
# MAX_TOKENIZATION_LENGTH = 512


transform = 'Normal'
if args.dataset == 'rafdb':
    train_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    test_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    task = 'emotion'
elif args.dataset == 'ferplus':
    train_dataset = FERPLUS(input_directory=os.path.join('/data/jc/data/image/ferplus_hapi',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    test_dataset = FERPLUS(input_directory=os.path.join('/data/jc/data/image/ferplus_hapi',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    task = 'emotion'
elif args.dataset == 'kdef':
    train_dataset = KDEF(input_directory=os.path.join('/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
    test_dataset = KDEF(input_directory=os.path.join('/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit',"valid"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform=transform)
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
    log_dir = 'optim_new/'+args.log_dir
    
if 'resnet' in args.model:
    model = getattr(models,'resnet')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'xlnet' in args.model:
    model = getattr(models,'xlnet')(model_name=args.model,num_classes=args.num_classes)
elif 'vit' in args.model:
    model = getattr(models,'vit')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'roberta' in args.model:
    model = getattr(models,'roberta')(model_name=args.model,num_classes=args.num_classes)
elif 't5' in args.model:
    model = getattr(models,'t5')(model_name=args.model,num_classes=args.num_classes)
elif 'vgg' in args.model:
    model = getattr(models,'vgg')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'efficientnet' in args.model:
    model = getattr(models,'efficientnet')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'alexnet' in args.model:
    model = getattr(models,'alexnet')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'googlenet' in args.model:
    model = getattr(models,'googlenet')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'densenet' in args.model:
    model = getattr(models,'densenet')(norm_par=train_dataset.norm_par,model_name=args.model,num_classes=args.num_classes)
elif 'autoencoder' in args.model:
    model = getattr(models,'autoencoder')(norm_par=train_dataset.norm_par)
if args.api == 'cifar10': 
    tea_model =  getattr(models,'resnet')(norm_par=train_dataset.norm_par,model_name='resnet50',num_classes=args.num_classes)    
    tea_model.load_state_dict(torch.load('/home/jkl6486/hermes/runs/cifar10gt/model.pth'))
else:
    tea_model = None
if parallel:
    model.model = nn.DataParallel(model.model).cuda()    
# model.load_state_dict(torch.load('/home/jkl6486/hermes/runs/fer_rafdb_facepp_fer_22-05-23_ep50_num_classes_7_lr0.0003_bs64_Lion_full_resnet50_percent_1.0_labeltrainfacepp_lr_schedulerdropout/model.pth'))
# Initialize train & test datasetsp

try:
    os.mkdir(log_dir)
except:
    print( "Directory %s already exists" % log_dir)


with open(os.path.join(log_dir,'args.txt'), mode='w') as f:
    json.dump(args.__dict__, f, indent=2)

args.G_activation = torch.tanh

generator = models.GeneratorA(nz=args.nz, nc=3, img_size=224, activation=args.G_activation)
generator = generator.cuda()
optimizer_G = torch.optim.Adam( generator.parameters(), lr=args.lr_G )

lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epochs)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,drop_last=True)


    
total_iters = len(train_dataset)/args.batch_size*args.epochs
    
optimizer_S, lr_scheduler_S = model.define_optimizer(
        parameters=args.op_parameters,
        OptimType=args.optimizer,
        lr=args.lr,custom_lr=args.custom_lr,weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        epochs=args.epochs, 
        lr_warmup_percent=args.lr_warmup_percent, 
        lr_warmup_epoch=args.lr_warmup_epoch,
        betas=args.betas,
        eps=args.eps,total_iters=total_iters)


 

    



    # for key, value in vars(args).items():
    #     f.write('%s:%s\n'%(key, str(value)))
    #     print(key, value) 
# parser = ArgumentParser()
# args = parser.parse_args()
# with open('commandline_args.txt', 'r') as f:
#     args.__dict__ = json.load(f)                  
distillation(module=model,num_classes=args.num_classes,
             epochs=args.epochs,lr_scheduler_S=lr_scheduler_S, lr_scheduler_G=lr_scheduler_G,
             log_dir=log_dir,grad_clip=args.grad_clip,
             validate_interval=args.validate_interval,save=args.save,
             
             loader_valid=test_loader,
           
              lr_scheduler_freq=args.lr_scheduler_freq,
             api=args.api,task=task,
            label_train=args.label_train,hapi_label_train=args.hapi_label_train,
             batch_size=args.batch_size,num_workers=args.num_workers,
             tea_model = tea_model,
             
             generator=generator,grad_epsilon=args.grad_epsilon,grad_m=args.grad_m,loss_type=args.loss_type,
                    epoch_itrs=args.epoch_itrs,nz=args.nz,optimizer_G=optimizer_G,optimizer_S=optimizer_S,
                    g_iter=args.g_iter,d_iter=args.d_iter,approx_grad=args.approx_grad,rec_grad_norm=args.rec_grad_norm,logit_correction=args.logit_correction,no_logits=args.logit_correction,
                 G_activation=args.G_activation,query_budget=args.query_budget,
                 forward_differences=args.forward_differences
                 
             )
