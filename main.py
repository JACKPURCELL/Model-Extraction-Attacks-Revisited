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

from dataset.imdb import IMDB
from dataset.yelp import YELP
from dataset.rafdb import RAFDB
from dataset.ferplus import FERPLUS
from dataset.kdef import KDEF
from dataset.cifar import CIFAR10
from utils.model import distillation
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
parser.add_argument('--fixmatch', action='store_true')
parser.add_argument('--usl', action='store_true')
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
parser.add_argument('--mu', default=3, type=int,
                        help='coefficient of unlabeled batch size')
parser.add_argument('--unlabel_batch', type=int, default=-1)
parser.add_argument('--label_batch', type=int, default=-1)

parser.add_argument('--pgd_percent', type=float)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--adaptive', choices=['entropy', 'kcenter','random'])
parser.add_argument('--n_samples', type=int)
parser.add_argument('--sample_times', type=int)

parser.add_argument('--api', type=str)
parser.add_argument('--lr_scheduler_freq', type=str,default='epoch')
parser.add_argument('--adv_train', choices=[None, 'pgd', 'free', 'cw'],
                           help='adversarial training (default: None)')
parser.add_argument('--adv_valid', choices=[None, 'pgd', 'free', 'cw'])
parser.add_argument('--encoder_train', action='store_true')
parser.add_argument('--encoder_path',  type=str)
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

if args.optimizer == 'Lion' and (args.dataset == 'imdb' or args.dataset == 'yelp'):
    args.betas = (0.95, 0.98)
elif args.dataset == 'imdb' or args.dataset == 'yelp':
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
    log_dir = 'submit2/cifar10/'+args.log_dir
    
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
elif args.dataset == 'yelp':
    path =  '/data/jc/data/sentiment/YELP/'
    train_dataset = YELP(input_directory=os.path.join(path,"train"),tokenizer=model.tokenizer,retokenize=args.retokenize,api=args.api,max_length=args.max_length,log_dir=args.log_dir)
    test_dataset = YELP(input_directory=os.path.join(path,"valid"),tokenizer=model.tokenizer,retokenize=args.retokenize,api=args.api,max_length=args.max_length,log_dir=args.log_dir)
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
if args.usl and (args.mixmatch or args.fixmatch):
    # selected_inds=[  3376,  8332,  8158,  4651, 10665, 10873,    63,  6225,  7267,
    #      704,  9358,  7070,  8233,  5628,  9265,  5855,  8803, 11265,
    #      445,  5462,  5644,  9034,   633,  7308,  4500,  1802, 11041,
    #     3848,  8908, 10100, 10763,  7740,  6941,  8359,  9888, 10064,
    #     2911,   566,  9957,  3965, 11923,  7125,  9171, 10386,  5434,
    #     9869,  5002,  6913,  2666,  1876, 12122,  5061, 10945,  7978,
    #     5470,  3982,  9043,  3470, 10937, 10554,  6342,  7910,  5804,
    #     3119,  5649,  5955,  9979,  2991,  6496,  3319,   251, 10982,
    #      729,  3611,  3333,  4051, 10239, 11604,  2475,  8478,   396,
    #     5541, 12066,  9512, 11973,  7805,  5656,  7455,  1358,  9665,
    #     5076,   692, 11267,  4398,   956, 11003,  2098,  1173,  8567,
    #     9191,  6416, 11537,  4033,   249,   684,  8340, 11758, 12185,
    #      726,   859,  4287, 11886,  7734,  4001,  8084, 10018,  3727,
    #     4465,   425,  4853,  9506,  8140, 10083,  7649,  8446,  7902,
    #     6610, 11828,  3243,  8406,  4230,   450, 10541,  5791,  7862,
    #       60,  9312,  7467, 11005, 10428,  6083,  3992, 11171, 10528,
    #     6983,  3548,  6561,  6321,  1663,  3136,  3308, 10706,  7366,
    #     5785,  8496,  4058,  5979,  3495,  9672, 11359,  5577, 10998,
    #     7576,  7183,  4958, 11630, 10086,  7198, 12209,  7983, 10270,
    #    12014, 11254,   889,  5502,  5547, 10369,  5398,  4134,  3125,
    #     1208,  5039, 12096,  8066,  9392,  7785,  7016,  8679,  2847,
    #     3643,  5754,  1765,  2996,  4594,  9090,  3857, 11529,   479,
    #     6073,  3337,   157,   955, 10166,  2670,  5326, 11451, 11388,
    #     4569,  1850,  8851, 12013, 11456,  1331,  2001, 10372,  2055,
    #     6461, 11139,  8125,  4667,  6471,  5553,  2516,  9021,  6502,
    #     2800,  2929,  7260,   585,  3451,   357,  7032,  9145, 10919,
    #    10424, 10443, 11785,  3004,  2377, 11869,   820,  1053, 11661,
    #      945, 11016,  9649,  2264,  8162,  9925,   187,  5287,  9198,
    #    12009,  5170,  3891,  7868]
    selected_inds=[8020,  4228,  8881,  1054,  1097,  4116,   182,  4873, 10462,
        1595,   137, 11866,   367,  2814, 10317,  4372, 11641,  8342,
        3086,   759,   906,  3273,  4059,  6186, 10980, 10087,  3484,
        3622,    44,  8974,  5036,  9636,  7166, 11505,  1596, 10151,
        9810,  5688,  1510,  2343,   416,  8171,  7277,  7606,  8672,
        6902, 10233, 11818, 11164,  4377,  1646,  7956,  6167,  6143,
        9186, 10277,   789,   768,  8537,  7014,   977,   509,  8820,
       11195, 10835,  2172,  1468,  2870,  8667,  4046,   674,  5411,
        5392,  8871,  3774,  9057, 10667,  3477,  1958,  9381,  8230,
        2404,  9733,  1981,  8245,  3734,   729,  3180,  9388,  6571,
        2754,  5711,  2960, 12187,   943, 10026,  3567,  7865, 11301,
       11947,  1757,   613,  8725,  5810,  1369,  9921,  1882,  9529,
        7591,  1254,  9125,  8052,  9533, 11965,  4214,  6667,  8356,
        4331,  1356,  2367, 11015,  9000,   857,  5452,  6924,  7696,
        6059,  7188,  9479,  6385,  2016,  9279,  1398,  3555,  1739,
        9570,    87,  5257,  6948,  8278, 10783,  6090,  7787, 11309,
        3278,  9732, 11477,  5626,  6587,  1329,   300,  6528,  7327,
        3383,  7162,  9258,  4401,   707, 11345,  4717, 11761,  4593,
        5892, 11013,  6474,  3357,  4986,  3073,  9782,  7755,  6728,
        6769,  9490,  7850, 10964,  4703,   714,  4951,  6479,  6864,
        6418,  5586,  1773,  8349,  9793, 10355,  5268,  9588,  3058,
       11137, 10365,  2442,  5696,  4198,  3013,  6910,  9646,  8478,
        4431,   985,  5100,  5624,  3262,  5299, 11591,  9131,  9416,
         213, 11095,  3232,  9507,    48,  6004,  5231,  5356,  1215,
        7684, 11922, 10876,  8589,  8588,  8261,  2657,  6137,   174,
        9623,  1735,  2788, 10494,  8057,  2892,  4687, 10542,  3185,
       11583,  3918,  8885,  1826,  7273, 12020, 11317, 10622,  1490,
        7539,  7170, 11437,  7285, 10366,  8161,  9041,  3641,  1706,
        6033,  9179,  4596,  7560]
    _label_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),
                           hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'Normal'),
                                selected_inds)

    new_index = np.arange(len(train_dataset))[~np.in1d(np.arange(len(train_dataset)),selected_inds)]
    np.random.shuffle(new_index)
    if args.mixmatch:
        transform_type = 'mixmatch'
    elif args.fixmatch:
        transform_type = 'fixmatch'
    _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),
                           hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = transform_type),
                                new_index)
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
    batch_size_unlabel = args.batch_size*args.mu if args.fixmatch else args.batch_size
    _unlabel_dataloader = DataLoader(dataset=_unlabel_dataset,
                    batch_size=batch_size_unlabel,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = itertools.cycle(_unlabel_dataloader)
    
elif args.mixmatch or args.fixmatch:#mixmatch
    _label_dataset, _unlabel_dataset = split_dataset(
        train_dataset,
        length=args.label_batch*args.batch_size,
        same_distribution=False,num_classes=args.num_classes,
        labels=train_dataset.targets)
    if args.mixmatch:
        transform_type = 'mixmatch'
    elif args.fixmatch:
        transform_type = 'fixmatch'
    _unlabel_dataset_indices = _unlabel_dataset.indices
    
    # _label_dataset, _temp_unlabel_dataset = split_dataset(
    #     _temp_dataset,length=args.label_batch*args.batch_size)

    if args.dataset == 'rafdb':
        _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = transform_type),
                                _unlabel_dataset_indices)
    elif args.dataset == 'ferplus':
        _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/ferplus_hapi',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = transform_type),
                                _unlabel_dataset_indices)
    elif args.dataset == 'kdef':
        _unlabel_dataset = Subset(KDEF(input_directory=os.path.join('/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = transform_type),
                                _unlabel_dataset_indices)
    elif args.dataset == 'cifar10':
        _unlabel_dataset = Subset(CIFAR10(mode='train',transform = transform_type),
                                _unlabel_dataset_indices)
    elif args.dataset == 'expw':
        _unlabel_dataset = Subset(EXPW(input_directory=os.path.join('/data/jc/data/image/EXPW_224',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = transform_type),
                                _unlabel_dataset_indices)
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
    batch_size_unlabel = args.batch_size*args.mu if args.fixmatch else args.batch_size
    
    _unlabel_dataloader = DataLoader(dataset=_unlabel_dataset,
                    batch_size=batch_size_unlabel,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = itertools.cycle(_unlabel_dataloader)
    
    
elif args.label_batch != -1:#adaptive or part data
    if args.usl:
        selected_inds=[8020,  4228,  8881,  1054,  1097,  4116,   182,  4873, 10462,
        1595,   137, 11866,   367,  2814, 10317,  4372, 11641,  8342,
        3086,   759,   906,  3273,  4059,  6186, 10980, 10087,  3484,
        3622,    44,  8974,  5036,  9636,  7166, 11505,  1596, 10151,
        9810,  5688,  1510,  2343,   416,  8171,  7277,  7606,  8672,
        6902, 10233, 11818, 11164,  4377,  1646,  7956,  6167,  6143,
        9186, 10277,   789,   768,  8537,  7014,   977,   509,  8820,
       11195, 10835,  2172,  1468,  2870,  8667,  4046,   674,  5411,
        5392,  8871,  3774,  9057, 10667,  3477,  1958,  9381,  8230,
        2404,  9733,  1981,  8245,  3734,   729,  3180,  9388,  6571,
        2754,  5711,  2960, 12187,   943, 10026,  3567,  7865, 11301,
       11947,  1757,   613,  8725,  5810,  1369,  9921,  1882,  9529,
        7591,  1254,  9125,  8052,  9533, 11965,  4214,  6667,  8356,
        4331,  1356,  2367, 11015,  9000,   857,  5452,  6924,  7696,
        6059,  7188,  9479,  6385,  2016,  9279,  1398,  3555,  1739,
        9570,    87,  5257,  6948,  8278, 10783,  6090,  7787, 11309,
        3278,  9732, 11477,  5626,  6587,  1329,   300,  6528,  7327,
        3383,  7162,  9258,  4401,   707, 11345,  4717, 11761,  4593,
        5892, 11013,  6474,  3357,  4986,  3073,  9782,  7755,  6728,
        6769,  9490,  7850, 10964,  4703,   714,  4951,  6479,  6864,
        6418,  5586,  1773,  8349,  9793, 10355,  5268,  9588,  3058,
       11137, 10365,  2442,  5696,  4198,  3013,  6910,  9646,  8478,
        4431,   985,  5100,  5624,  3262,  5299, 11591,  9131,  9416,
         213, 11095,  3232,  9507,    48,  6004,  5231,  5356,  1215,
        7684, 11922, 10876,  8589,  8588,  8261,  2657,  6137,   174,
        9623,  1735,  2788, 10494,  8057,  2892,  4687, 10542,  3185,
       11583,  3918,  8885,  1826,  7273, 12020, 11317, 10622,  1490,
        7539,  7170, 11437,  7285, 10366,  8161,  9041,  3641,  1706,
        6033,  9179,  4596,  7560]
        _label_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),
                            hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'Normal'),
                                    selected_inds)

        unlabel_dataset_indices = np.arange(len(train_dataset))[~np.in1d(np.arange(len(train_dataset)),selected_inds)]
        np.random.shuffle(unlabel_dataset_indices)
        _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),
                            hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'mixmatch'),
                                    unlabel_dataset_indices)
    else:
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
    f.write("epoch,gt_acc1,hapi_loss,hapi_acc1\n")

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
             mixmatch=args.mixmatch,mu=args.mu,fixmatch=args.fixmatch,
             save=args.save,label_train=args.label_train,
             hapi_label_train=args.hapi_label_train, lr_scheduler_freq=args.lr_scheduler_freq,
             api=args.api,task=task,unlabel_dataset_indices=unlabel_dataset_indices,
             hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,
             batch_size=args.batch_size,num_workers=args.num_workers,
             n_samples = args.n_samples,adaptive=args.adaptive,get_sampler_fn=get_sampler,
             balance=args.balance,sample_times=args.sample_times,tea_model=tea_model,AE=AE,encoder_attack=args.encoder_attack,pgd_percent=args.pgd_percent,
             encoder_train=args.encoder_train,train_dataset=train_dataset,workers=args.num_workers)
