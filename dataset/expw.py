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
# hapi.config.data_dir = "/home/ljc/HAPI" 


def get_transform_base(mode: str, use_tuple: bool = False,
                           auto_augment: bool = False, crop_shape = 224,norm_par=None) -> transforms.Compose:
    if mode == 'train':
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
            # transforms.Resize((crop_shape, crop_shape) if use_tuple else crop_shape),
            # transforms.CenterCrop((crop_shape, crop_shape) if use_tuple else crop_shape),
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
       
       
class EXPW(datasets.ImageFolder):
    
    def __init__(self, input_directory=None, hapi_data_dir:str = None, hapi_info:str = None, api=None,transform='Normal'):


        mode = input_directory.split('/')[-1]
        if transform == 'Normal':
            transform = get_transform_base(
                mode, use_tuple=True,
                auto_augment=True, crop_shape = 224,
                norm_par=None)
        elif transform == 'mixmatch':
            transform = get_transform_base(
                mode, use_tuple=True,
                auto_augment=True, crop_shape = 224,
                norm_par=None)
            transform = TransformTwice(transform)
        elif transform == 'raw':
            transform = transforms.Compose([transforms.PILToTensor(),
                                   transforms.ConvertImageDtype(torch.float)])
        else:
            transform_list=[]
            transform_list.append(transforms.PILToTensor())
            transform_list.append(transforms.ConvertImageDtype(torch.float))
            transform = transforms.Compose(transform_list)
        super().__init__(root=input_directory,transform=transform)
        hapi.config.data_dir = hapi_data_dir
        self.api = api
        self.norm_par = {'mean': [0.579, 0.454, 0.399],'std': [0.275, 0.249, 0.245]}
        
        dic = hapi_info 

        dic_split = dic.split('/')
        predictions =  hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])

        self.info_lb = torch.zeros(40000,dtype=torch.long)
        self.info_conf = torch.zeros(40000)

        
        for i in range(len(predictions[dic])):
            hapi_id = int(predictions[dic][i]['example_id'].split('.')[0])
            self.info_lb[hapi_id] = torch.tensor((predictions[dic][i]['predicted_label']))
            self.info_conf[hapi_id] = torch.tensor((predictions[dic][i]['confidence']))

    
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
                #TODO: add amazon api
                with open(os.path.join('/data/jc/data/image/RAFDB', 'amazon_api', path.split('/')[-1]), mode='r') as p:
                    api_result = json.load(p)
            case 'facepp':

                #TODO: add facepp api
                
                with open(os.path.join('/data/jc/data/image/RAFDB', 'facepp_api', path.split('/')[-1]), mode='r') as p:
                    api_result = json.load(p)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        match self.api:
            case 'facepp':
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
                else:
                    soft_label = torch.ones(7)*0.14285714285714285
                    hapi_label = torch.tensor(6)
            case 'amazon':
                soft_label = torch.ones(7)
                if len(api_result[0]) != 1:
                    soft_label[0] = api_result[0]['ANGRY']*0.01
                    soft_label[1] = api_result[0]['DISGUSTED']*0.01
                    soft_label[2] = api_result[0]['FEAR']*0.01
                    soft_label[3] = api_result[0]['HAPPY']*0.01
                    soft_label[4] = api_result[0]['SAD']*0.01
                    soft_label[5] = api_result[0]['SURPRISED']*0.01
                    soft_label[6] = api_result[0]['CALM']*0.01 + api_result[0]['CONFUSED']*0.01
                    # soft_label[7] = api_result[0]['CONFUSED']

                    hapi_label = torch.argmax(soft_label)
                else:
                    soft_label = torch.ones(7)*0.14285714285714285
                    hapi_label = torch.tensor(6)  
            case _:
                
                hapi_id = torch.tensor(int(path.split('/')[-1].split('.')[0]))
                hapi_label = self.info_lb[hapi_id]
                hapi_confidence = self.info_conf[hapi_id]
                other_confidence = (1 - hapi_confidence)/6
                soft_label = torch.ones(7)*other_confidence
                soft_label[int(hapi_label)] = hapi_confidence

        
        return sample, target, soft_label, hapi_label
    
