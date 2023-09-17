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
                           auto_augment: bool = False, crop_shape = 224,norm_par=None) -> transforms.Compose:
    if mode == 'train':
        transform_list = [
            transforms.RandomResizedCrop((crop_shape, crop_shape) if use_tuple else crop_shape),
            transforms.RandomHorizontalFlip(),
        ]
        # transform_list = [
        #     transforms.RandomRotation(90),
        #     transforms.Grayscale(num_output_channels=3),
        # ]
    
      
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
       
class KDEF(datasets.ImageFolder):
    def __init__(self, input_directory=None, hapi_data_dir:str = None, hapi_info:str = None, api=None,transform='Normal'):
#         0.5084671 0.3032051 0.2204921
# 0.2170966 0.16445793 0.12108668
        mode = input_directory.split('/')[-1]
        if mode == 'valid':
            mode = 'test'
        if transform == 'Normal':
            transform = get_transform_base(
                mode, use_tuple=False,
                auto_augment=True, crop_shape = 224)
                # norm_par=None)
                
                # norm_par={'mean': [0.509, 0.303, 0.221],'std': [0.217, 0.164, 0.121]})
            # transform =transforms.Compose([

            #                         transforms.PILToTensor(),
            #                         transforms.ConvertImageDtype(torch.float)])
        elif transform == 'mixmatch':
            transform = get_transform_base(
                mode, use_tuple=False,
                auto_augment=True, crop_shape = 224)
            #    norm_par={'mean': [0.509, 0.303, 0.221],'std': [0.217, 0.164, 0.121]})
            transform = TransformTwice(transform)
        elif transform == 'raw':
            transform = transforms.Compose([transforms.Resize(224),
                                            transforms.PILToTensor(),
                                   transforms.ConvertImageDtype(torch.float)])
        super().__init__(root=input_directory,transform=transform)
        self.norm_par = {'mean': [0.509, 0.303, 0.221],'std': [0.217, 0.164, 0.121]}
        self.api = api


    
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
                with open(os.path.join('/data/jc/data/image/KDEF_and_AKDEF', 'amazon_api', path.split('/')[-1]), mode='r') as p:
                    api_result = json.load(p)
            case 'facepp':
                with open(os.path.join('/data/jc/data/image/KDEF_and_AKDEF', 'facepp_api', path.split('/')[-1]), mode='r') as p:
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
                quanti = True
                
                soft_label = torch.ones(8)
                if len(api_result[0]) != 1:
                    soft_label[0] = api_result[0]['ANGRY']*0.01
                    soft_label[1] = api_result[0]['DISGUSTED']*0.01
                    soft_label[2] = api_result[0]['FEAR']*0.01
                    soft_label[3] = api_result[0]['HAPPY']*0.01
                    soft_label[4] = api_result[0]['SAD']*0.01
                    soft_label[5] = api_result[0]['SURPRISED']*0.01
                    soft_label[6] = api_result[0]['CALM']*0.01 
                    soft_label[7] = api_result[0]['CONFUSED']*0.01

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
                raise ValueError('API is not supported')
                hapi_id = torch.tensor(int(re.findall(r'_(.*).jpg', path)[0]))
                hapi_label = self.info_lb[hapi_id]
                hapi_confidence = self.info_conf[hapi_id]
                other_confidence = (1 - hapi_confidence)/6
                soft_label = torch.ones(7)*other_confidence
                soft_label[int(hapi_label)] = hapi_confidence

        
        return sample, target, soft_label, hapi_label
    
