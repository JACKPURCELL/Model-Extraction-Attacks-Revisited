# -*- coding: utf-8 -*-
import argparse
import json
import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms
import torchvision.datasets as datasets
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
import re

import hapi

def get_transform_base(mode: str, use_tuple: bool = False,
                           auto_augment: bool = False, crop_shape = 100,norm_par=None) -> transforms.Compose:
    if mode == 'train':
        # transform_list = [
        #     transforms.RandomRotation(90),
        #     transforms.Grayscale(num_output_channels=3),
        # ]
        transform_list = [
            transforms.RandomResizedCrop((crop_shape, crop_shape) if use_tuple else crop_shape),
            transforms.RandomHorizontalFlip(),
        ]
        # transform_list=[]
        if auto_augment:
            transform_list.append(transforms.AutoAugment(
                transforms.AutoAugmentPolicy.IMAGENET))
        transform_list.append(transforms.PILToTensor())
        transform_list.append(transforms.ConvertImageDtype(torch.float))
        transform = transforms.Compose(transform_list)
    else:
        # TODO: torchvision.prototypes.transforms._presets.ImageClassificationEval
        transform = transforms.Compose([
            transforms.Resize((crop_shape, crop_shape) if use_tuple else crop_shape),
            transforms.CenterCrop((crop_shape, crop_shape) if use_tuple else crop_shape),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)])
    if norm_par is not None:
        transform.transforms.append(transforms.Normalize(
        mean=norm_par['mean'], std=norm_par['std']))
    return transform

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

# from .randaugment import RandAugmentMC

    
class TransformFixMatch(object):
    def __init__(self):
        
        
        self.weak = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect')]
            )
        self.strong = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.AutoAugment(
                transforms.AutoAugmentPolicy.IMAGENET),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)]
            )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    
def quantize_number(num):
    if num == 1/7 or num == 1/8:
        return num
        
    if num >= 0.9:
        q_num =1
    elif num >= 0.7:
        q_num =0.8
    elif num >= 0.5:
        q_num =0.6
    elif num >= 0.3:
        q_num =0.4
    elif num>= 0.125:
        q_num =0.2
    else:
        q_num =0    
    return torch.tensor(q_num)
    # return torch.round((num + 0.1) / 0.2) * 0.2 - 0.1           
class FERPLUS(datasets.ImageFolder):
    
    def __init__(self, input_directory=None, hapi_data_dir:str = None, hapi_info:str = None, api=None,transform='Normal'):
        mode = input_directory.split('/')[-1]
        if mode == 'valid':
            mode = 'test'

        if transform == 'Normal':
            transform = get_transform_base(
                mode, use_tuple=False,
                auto_augment=False, crop_shape = 224,
                norm_par=None)
        elif transform == 'mixmatch':
            transform = get_transform_base(
                mode, use_tuple=False,
                auto_augment=False, crop_shape = 224,
                norm_par=None)
            transform = TransformTwice(transform)
        elif transform == 'fixmatch':
            transform = TransformFixMatch()
        elif transform == 'fixmatch_Normal':
            transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                padding=int(224*0.125),
                                padding_mode='reflect'),
            transforms.AutoAugment(
                transforms.AutoAugmentPolicy.IMAGENET),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        elif transform == 'raw':
            transform = transforms.Compose([transforms.Resize(224),
                                            transforms.PILToTensor(),
                                   transforms.ConvertImageDtype(torch.float)])
        else:
            transform_list=[]
            transform_list.append(transforms.PILToTensor())
            transform_list.append(transforms.ConvertImageDtype(torch.float))
            transform = transforms.Compose(transform_list)
        super().__init__(root=input_directory,transform=transform)
        hapi.config.data_dir = hapi_data_dir
        self.api = api

        # self.norm_par = {'mean': [0.485, 0.456, 0.406],'std': [0.229, 0.224, 0.225]}
        self.norm_par = {'mean': [0.507,0.507, 0.507],'std': [0.250, 0.250, 0.250]}
        dic = hapi_info 

        dic_split = dic.split('/')
        predictions =  hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])

        self.info_lb = torch.zeros(100000,dtype=torch.long)
        self.info_conf = torch.zeros(100000)

        for i in range(len(predictions[dic])):
            hapi_id = int(predictions[dic][i]['example_id'].split('fer')[1])

            self.info_lb[hapi_id] = torch.tensor((predictions[dic][i]['predicted_label']))
            temp = predictions[dic][i]['confidence']
            
            self.info_conf[hapi_id] = torch.tensor((temp))



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        match self.api:
            case 'amazon':
                with open(os.path.join('/data/jc/data/image/ferplus_hapi', 'amazon_api', path.split('/')[-1]), mode='r') as p:
                    api_result = json.load(p)
            case 'facepp':
                with open(os.path.join('/data/jc/data/image/ferplus_hapi', 'facepp_api', path.split('/')[-1]), mode='r') as p:
                    api_result = json.load(p)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        match self.api:
            case 'facepp':
                quanti = True
                soft_label = torch.ones(7)
                if len(api_result[0]) != 1:
                    soft_label[0] = api_result[0]['anger']*0.01
                    soft_label[1] = api_result[0]['disgust']*0.01
                    soft_label[2] = api_result[0]['fear']*0.01
                    soft_label[3] = api_result[0]['happiness']*0.01
                    soft_label[4] = api_result[0]['sadness']*0.01
                    soft_label[5] = api_result[0]['surprise']*0.01
                    soft_label[6] = api_result[0]['neutral']*0.01 
                    # soft_label[7] = api_result[0]['CONFUSED']

                    hapi_label = torch.argmax(soft_label)
                    if quanti:
                        hapi_confidence = soft_label[int(hapi_label)]
                        hapi_confidence = quantize_number(hapi_confidence)
                        other_confidence = (1 - hapi_confidence)/6
                        soft_label = torch.ones(7)*other_confidence
                        soft_label[int(hapi_label)] = hapi_confidence
                        
                else:
                    soft_label = torch.ones(7)*0.14285714285714285
                    hapi_label = torch.tensor(6)
            case 'amazon':
                soft_label = torch.ones(8)
                quanti = True
                
                if len(api_result[0]) != 1:
                    soft_label[0] = api_result[0]['ANGRY']*0.01
                    soft_label[1] = api_result[0]['DISGUSTED']*0.01
                    soft_label[2] = api_result[0]['FEAR']*0.01
                    soft_label[3] = api_result[0]['HAPPY']*0.01
                    soft_label[4] = api_result[0]['SAD']*0.01
                    soft_label[5] = api_result[0]['SURPRISED']*0.01
                    soft_label[6] = api_result[0]['CALM']*0.01
                    soft_label[7] = api_result[0]['CONFUSED'] *0.01

                    hapi_label = torch.argmax(soft_label)
                    if quanti:
                        hapi_confidence = quantize_number(soft_label[int(hapi_label)])
                        other_confidence = (1 - hapi_confidence)/7
                        soft_label = torch.ones(8)*other_confidence
                        soft_label[int(hapi_label)] = hapi_confidence
                else:
                    soft_label = torch.ones(8)*0.125
                    hapi_label = torch.tensor(7)        
  
            # case 'microsoft':
            #     soft_label = torch.ones(3)
            #     soft_label[0] = api_result[0]['positive']
            #     soft_label[1] = api_result[0]['negative']
            #     soft_label[2] = api_result[0]['neutral']
            #     if soft_label[0] >= soft_label[1]:
            #         hapi_label = torch.tensor(0)
            #     else:
            #         hapi_label = torch.tensor(1)
            case _:
                hapi_id = torch.tensor(int(re.findall(r'/fer0(.*).png', path)[0]))
                hapi_label = self.info_lb[hapi_id]
                hapi_confidence = self.info_conf[hapi_id]
                other_confidence = (1 - hapi_confidence)/6
                soft_label = torch.ones(7)*other_confidence
                soft_label[int(hapi_label)] = hapi_confidence

        
        return sample, target, soft_label, hapi_label
  