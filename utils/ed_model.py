"""
Script containing various utilities related to model training, testing, and extraction of attention
weights.
"""

import base64
from collections import OrderedDict
import io
import json
import logging
import os
from typing import Callable, Iterable, Iterator
import imageio
import requests
import tensorboard
import torch.nn as nn
import torch
import boto3



client = boto3.client('rekognition')

import torch.nn.functional as F
import numpy as np


from optimizer.optim import Optimizer
from utils.tensor import add_noise
from .logger import MetricLogger

from .output import ansi, get_ansi_len, output_iter, prints
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
from PIL import Image

"""
Script for training, testing, and saving finetuned, binary classification models based on pretrained
BERT parameters, for the IMDB dataset.
"""

import itertools
from torch.utils.data import DataLoader
from torchattacks import CW
from torchattacks import PGD

# !pip install pytorch_transformers

from dataset.imdb import IMDB
from dataset.rafdb import RAFDB
from torch.utils.data import Dataset,Subset

from torch.utils.tensorboard import SummaryWriter


def loss_fn(_input: torch.Tensor = None, _label: torch.Tensor = None,
            _output: torch.Tensor = None, reduction: str = 'mean',_soft_label: torch.Tensor = None,
            temp: float = 1.0, outputs_x=None, targets_x=None, outputs_u=None, targets_u=None, iter=None,
            **kwargs) -> torch.Tensor:
    r"""Calculate the loss using :attr:`self.criterion`
    (:attr:`self.criterion_noreduction`).

    Args:
        _input (torch.Tensor | None): The batched input tensor.
            If :attr:`_output` is provided, this argument will be ignored.
            Defaults to ``None``.
        _label (torch.Tensor): The label of the batch with shape ``(N)``.
        _output (torch.Tensor | None): The logits of :attr:`_input`.
            If ``None``, use :attr:`_input` to calculate logits.
            Defaults to ``None``.
        reduction (str): Specifies the reduction to apply to the output.
            Choose from ``['none', 'mean']``.
            Defaults to ``'mean'``.
        **kwargs: Keyword arguments passed to :meth:`get_logits()`
            if :attr:`_output` is not provided.

    Returns:
        torch.Tensor:
            A scalar loss tensor (with shape ``(N)`` if ``reduction='none'``).
    """
    if outputs_x is not None:
        return SemiLoss(outputs_x, targets_x, outputs_u, targets_u, iter)
        
    if _soft_label is None:
        return val_loss(_input=_input, _label=_label, _output=_output, reduction=reduction)
        
    criterion = nn.CrossEntropyLoss(reduction=reduction)

    # return criterion(_output,_soft_label)+torch.mean((_output - _soft_label)**2)*100
    return criterion(_output,_soft_label)

@torch.no_grad()
def save_fn( log_dir,module, verbose: bool = False, indent: int = 0):
    r"""Save pretrained model weights.

    Args:
        file_path (str | None): The file path to save pretrained weights.
            Defaults to ``'{folder_path}/{self.name}{suffix}.pth'``.
        folder_path (str | None): The folder path containing model checkpoint.
            It is used when :attr:`file_path` is not provided.
            Defaults to :attr:`self.folder_path`.
        suffix (str | None): The suffix string to model weights file.
            Defaults to :attr:`self.suffix`.
        component (str): Specify which part of the weights to save.
            Choose from ``['full', 'features', 'classifier']``.
            Defaults to ``'full'``.
        verbose (bool): Whether to output auxiliary information.
            Defaults to ``False``.
        indent (int): The indent of output auxialiary information.
        **kwargs: Keyword arguments passed to :any:`torch.save`.
    """

    file_path = os.path.normpath(os.path.join(log_dir, f'model.pth'))

    _dict: OrderedDict[str, torch.Tensor] = module.state_dict()
    torch.save(_dict, file_path)
    if verbose:
        prints(
            f'Model saved at: {file_path}', indent=indent)
        
        
@torch.no_grad()
def accuracy_fn(_output: torch.Tensor, _label: torch.Tensor, num_classes: int,
             topk: Iterable[int] = (1, 5)) -> list[float]:
    r"""Computes the accuracy over the k top predictions
    for the specified values of k.

    Args:
        _output (torch.Tensor): The batched logit tensor with shape ``(N, C)``.
        _label (torch.Tensor): The batched label tensor with shape ``(N)``.
        num_classes (int): Number of classes.
        topk (~collections.abc.Iterable[int]): Which top-k accuracies to show.
            Defaults to ``(1, 5)``.

    Returns:
        list[float]: Top-k accuracies.
    """
    maxk = min(max(topk), num_classes)
    batch_size = _label.size(0)
    _, pred = _output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(_label[None])
    res: list[float] = []
    for k in topk:
        if k > num_classes:
            res.append(100.0)
        else:
            correct_k = float(correct[:k].sum(dtype=torch.float32))
            res.append(correct_k * (100.0 / batch_size))
    return res
    
    
def dis_validate(module: nn.Module, num_classes: int,
             loader: torch.utils.data.DataLoader,
             print_prefix: str = 'Validate', indent: int = 0,
             verbose: bool = True,
             writer=None, main_tag: str = 'valid',
             tag: str = '', _epoch: int = None,
             label_train = False, hapi_label_train=False,api=False,task=None,after_loss_fn=None,adv_valid=False,tea_model=None,
             **kwargs) -> tuple[float, float]:
    r"""Evaluate the model.

    Returns:
        (float, float): Accuracy and loss.
    """
    module.eval()
   
    forward_fn =  module.__call__

    logger = MetricLogger()
    # if api is not None:
    #     logger.create_meters(gt_loss=None, gt_acc1=None, 
    #                          hapi_loss=None, hapi_acc1=None)
    # else:
    #     # logger.create_meters( gt_loss=None, gt_acc1=None, 
    #     #                     hapi_loss=None, hapi_acc1=None,
    #     #                     tt=None,tf=None,ft=None,ff=None)
    #     logger.create_meters( gt_loss=None, gt_acc1=None, 
    #                         hapi_loss=None, hapi_acc1=None)
    if adv_valid:
        logger.create_meters(gt_loss=None, gt_acc1=None, 
                             hapi_loss=None, hapi_acc1=None,
                             adv_loss=None, adv_acc1=None)
    else:
        logger.create_meters(gt_loss=None, gt_acc1=None, 
                             hapi_loss=None, hapi_acc1=None)
    
    loader_epoch = loader  
    if verbose:
        header: str = '{yellow}{0}{reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))
        loader_epoch = logger.log_every(loader, header=header,
                                        tqdm_header='Batch',
                                        indent=indent)
    for data in loader_epoch:
        if adv_valid:
            match task:
                case 'emotion':

                    _input, _label, _soft_label, hapi_label  = data
                    _input = _input.cuda()
                    if tea_model is not None:
                        m = nn.Softmax(dim=1)
                        with torch.no_grad():
                            _soft_label=m(tea_model(_input))
                        hapi_label = torch.argmax(_soft_label, dim=-1)
                    _soft_label = _soft_label.cuda()
                    _label = _label.cuda()
                    hapi_label = hapi_label.cuda()
                    _output = forward_fn(_input)
                    if adv_valid:
                        loss,adv_x, adv_api_soft_label, adv_api_hapi_label = after_loss_fn(_input=_input,_label=_label,_output=_output,mode='valid')
                        adv_x = adv_x.cuda()
                        adv_api_soft_label = adv_api_soft_label.cuda()
                        adv_output = forward_fn(adv_x)
                        adv_loss = float(loss_fn(_soft_label=adv_api_soft_label, _output=adv_output,  **kwargs))
                        


                case 'sentiment':
                    # input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label  = data
                    input_ids, attention_mask, _label, _soft_label, hapi_label  = data
                    input_ids = input_ids.cuda()
                    # token_type_ids = token_type_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    _label = _label.cuda()
                    _soft_label = _soft_label.cuda()
                    hapi_label = hapi_label.cuda()

                    _output = forward_fn(input_ids=input_ids,attention_mask=attention_mask)
                    if adv_valid:
                        raise NotImplementedError(f'{adv_valid=} is not supported on sentiment yet.')
        
            gt_loss = float(loss_fn( _label=_label, _output=_output, **kwargs))
            if label_train:
                hapi_loss = float(loss_fn( _label=hapi_label, _output=_output,  **kwargs))
            elif hapi_label_train:
                hapi_loss = float(loss_fn( _label=hapi_label, _output=_output,  **kwargs))
            else:    
                hapi_loss = float(loss_fn( _soft_label=_soft_label, _output=_output,  **kwargs))



            batch_size = int(_label.size(0))
            match task:
                case 'sentiment':
                    _output = _output[:,:2]
                    new_num_classes = 2
                case 'emotion':
                    new_num_classes = num_classes
            hapi_acc1, hapi_acc5 = accuracy_fn(
                    _output, hapi_label, num_classes=new_num_classes, topk=(1, 5))
            gt_acc1, gt_acc5 = accuracy_fn(
                _output, _label, num_classes=new_num_classes, topk=(1, 5))
            if adv_valid:
                adv_acc1, adv_acc5 = accuracy_fn(
                    adv_output, adv_api_hapi_label, num_classes=new_num_classes, topk=(1, 5))
                
                logger.update(n=batch_size,  gt_loss=float(gt_loss), gt_acc1=gt_acc1, 
                            hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1,
                            adv_loss=adv_loss, adv_acc1=adv_acc1)
            else:    
                logger.update(n=batch_size,  gt_loss=float(gt_loss), gt_acc1=gt_acc1, 
                            hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1)
        else:
            with torch.no_grad():
                match task:
                    case 'emotion':

                        _input, _label, _soft_label, hapi_label  = data
                        _input = _input.cuda()
                        if tea_model is not None:
                            m = nn.Softmax(dim=1)
                            with torch.no_grad():
                                _soft_label=m(tea_model(_input))
                            hapi_label = torch.argmax(_soft_label, dim=-1)
                        _soft_label = _soft_label.cuda()
                        _label = _label.cuda()
                        hapi_label = hapi_label.cuda()
                        _output = forward_fn(_input)
                        
                            


                    case 'sentiment':
                        # input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label  = data
                        input_ids, attention_mask, _label, _soft_label, hapi_label  = data
                        input_ids = input_ids.cuda()
                        # token_type_ids = token_type_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        _label = _label.cuda()
                        _soft_label = _soft_label.cuda()
                        hapi_label = hapi_label.cuda()

                        _output = forward_fn(input_ids=input_ids,attention_mask=attention_mask)
                        if adv_valid:
                            raise NotImplementedError(f'{adv_valid=} is not supported on sentiment yet.')
                    case 'cifar10':
                        _input, _label  = data
                        _input = _input.cuda()
                        _label = _label.cuda()
                        hapi_label=_label
                        if tea_model is not None:
                            m = nn.Softmax(dim=1)
                            with torch.no_grad():
                                _soft_label=m(tea_model(_input))
                            hapi_label = torch.argmax(_soft_label, dim=-1)
                        _output = forward_fn(_input)

                        
                gt_loss = float(loss_fn( _label=_label, _output=_output, **kwargs))
                if label_train:
                    hapi_loss = float(loss_fn( _label=hapi_label, _output=_output,  **kwargs))
                elif hapi_label_train:
                    hapi_loss = float(loss_fn( _label=hapi_label, _output=_output,  **kwargs))
                else:    
                    hapi_loss = float(loss_fn( _soft_label=_soft_label, _output=_output,  **kwargs))



                batch_size = int(_label.size(0))
                match task:
                    case 'sentiment':
                        _output = _output[:,:2]
                        new_num_classes = 2
                    case 'emotion':
                        new_num_classes = num_classes
                    case 'cifar10':
                        new_num_classes = num_classes
                hapi_acc1, hapi_acc5 = accuracy_fn(
                        _output, hapi_label, num_classes=new_num_classes, topk=(1, 5))
                gt_acc1, gt_acc5 = accuracy_fn(
                    _output, _label, num_classes=new_num_classes, topk=(1, 5))
                if adv_valid:
                    adv_acc1, adv_acc5 = accuracy_fn(
                        adv_output, adv_api_hapi_label, num_classes=new_num_classes, topk=(1, 5))
                    
                    logger.update(n=batch_size,  gt_loss=float(gt_loss), gt_acc1=gt_acc1, 
                                hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1,
                                adv_loss=adv_loss, adv_acc1=adv_acc1)
                else:    
                    logger.update(n=batch_size,  gt_loss=float(gt_loss), gt_acc1=gt_acc1, 
                                hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1)
         
    if api is not None:
        gt_loss, gt_acc1, hapi_loss, hapi_acc1 = (logger.meters['gt_loss'].global_avg,
                    logger.meters['gt_acc1'].global_avg,
                 logger.meters['hapi_loss'].global_avg,
                 logger.meters['hapi_acc1'].global_avg)
        if writer is not None and _epoch is not None and main_tag:
            from torch.utils.tensorboard import SummaryWriter
            assert isinstance(writer, SummaryWriter)
            writer.add_scalars(main_tag='gt_loss/' + main_tag,
                        tag_scalar_dict={tag: gt_loss}, global_step=_epoch)
            writer.add_scalars(main_tag='gt_acc1/' + main_tag,
                        tag_scalar_dict={tag: gt_acc1}, global_step=_epoch) 
            writer.add_scalars(main_tag='hapi_loss/' + main_tag,
                        tag_scalar_dict={tag: hapi_loss}, global_step=_epoch)
            writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
                        tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch)

    else:   
        gt_loss, gt_acc1, hapi_loss, hapi_acc1,tt,tf,ft,ff = (logger.meters['gt_loss'].global_avg,
                    logger.meters['gt_acc1'].global_avg,
                    logger.meters['hapi_loss'].global_avg,
                    logger.meters['hapi_acc1'].global_avg,
                    logger.meters['tt'].global_avg,
                    logger.meters['tf'].global_avg,
                    logger.meters['ft'].global_avg,
                    logger.meters['ff'].global_avg)

        if writer is not None and _epoch is not None and main_tag:
            from torch.utils.tensorboard import SummaryWriter
            assert isinstance(writer, SummaryWriter)
            writer.add_scalars(main_tag='gt_loss/' + main_tag,
                        tag_scalar_dict={tag: gt_loss}, global_step=_epoch)
            writer.add_scalars(main_tag='gt_acc1/' + main_tag,
                        tag_scalar_dict={tag: gt_acc1}, global_step=_epoch)        
            writer.add_scalars(main_tag='hapi_loss/' + main_tag,
                        tag_scalar_dict={tag: hapi_loss}, global_step=_epoch)
            writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
                        tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch)
            writer.add_scalars(main_tag='tt/' + main_tag,
                            tag_scalar_dict={tag: tt}, global_step=_epoch)
            writer.add_scalars(main_tag='tf/' + main_tag,
                            tag_scalar_dict={tag: tf}, global_step=_epoch)
            writer.add_scalars(main_tag='ft/' + main_tag,
                            tag_scalar_dict={tag: ft}, global_step=_epoch)
            writer.add_scalars(main_tag='ff/' + main_tag,
                            tag_scalar_dict={tag: ff}, global_step=_epoch)

    return hapi_acc1, hapi_loss
def activate_params(module: nn.Module, params: Iterator[nn.Parameter] = []):
    r"""Set ``requires_grad=True`` for selected :attr:`params` of :attr:`module`.
    All other params are frozen.

    Args:
        module (torch.nn.Module): The module to process.
        params (~collections.abc.Iterator[torch.nn.parameter.Parameter]):
            The parameters to ``requires_grad``.
                Defaults to ``[]``.
    """
    module.requires_grad_(False)
    for param in params:
        param.requires_grad_()



def distillation(module: nn.Module, pgd_set,num_classes: int,
          epochs: int, optimizer, lr_scheduler,adv_train=None,adv_train_iter=7,adv_valid=False,
        log_dir:str = 'runs/test', 
          grad_clip: float = 5.0, 
          print_prefix: str = 'Distill', start_epoch: int = 0, resume: int = 0,
          validate_interval: int = 1, save: bool = True,
          loader_train: torch.utils.data.DataLoader = None,
          loader_valid: torch.utils.data.DataLoader = None,
          unlabel_iterator = None,
        file_path: str = None,
          folder_path: str = None, suffix: str = None,
           main_tag: str = 'train', tag: str = '',

          verbose: bool = True, output_freq: str = 'iter', indent: int = 0,
          change_train_eval: bool = True, lr_scheduler_freq: str = 'epoch',
          backward_and_step: bool = True, 
          mixmatch: bool = False,label_train: bool=False,hapi_label_train: bool=False,
          api=False,task='sentiment',unlabel_dataset_indices=None,
          hapi_data_dir=None,hapi_info=None,
        batch_size=None,num_workers=None,
        n_samples = None,adaptive=False,get_sampler_fn=None,
        balance=False,sample_times = 10,tea_model=None,
          pgd_percent=None,
          
          
          **kwargs):
    r"""Train the model"""
    if epochs <= 0:
        return
    after_loss_fn = None
    forward_fn =  module.__call__
    if adv_train is not None or adv_valid:
        if adv_train == 'pgd':
            pgd = PGD(module, eps=8/255,alpha=2/255, steps=20, random_start=True)
        elif adv_train == 'cw':
            cw = CW(module, c=1, kappa=0, steps=50, lr=0.01)
        else:
            raise NotImplementedError(f'{adv_train=} is not supported yet.')
        
        


    writer = SummaryWriter(log_dir=log_dir)
    validate_fn = dis_validate 


    scaler: torch.cuda.amp.GradScaler = None

    best_validate_result = (0.0, float('inf'))
    best_acc = 0.0

        

    params: list[nn.Parameter] = []
    for param_group in optimizer.param_groups:
        params.extend(param_group['params'])
    len_loader_train = len(loader_train)
    total_iter = (epochs - resume) * len_loader_train

    logger = MetricLogger()
    if mixmatch:
        logger.create_meters(loss=None)
    elif adv_train:
        logger.create_meters(   gt_acc1=None, 
                          hapi_loss=None, hapi_acc1=None,attack_succ=None,ahapi_succ=None)
    else:
        logger.create_meters(   gt_acc1=None, 
                          hapi_loss=None, hapi_acc1=None)
    if resume and lr_scheduler:
        for _ in range(resume):
            lr_scheduler.step()
    iterator = range(resume, epochs)
    if verbose and output_freq == 'epoch':
        header: str = '{blue_light}{0}: {reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(header), 30) + get_ansi_len(header))
        iterator = logger.log_every(range(resume, epochs),
                                    header=print_prefix,
                                    tqdm_header='Epoch',
                                    indent=indent)
    new_label_indices = None
    
    for _epoch in iterator:
        _epoch += 1
        logger.reset()
        loader_epoch = loader_train
        if verbose and output_freq == 'iter':
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            loader_epoch = logger.log_every(loader_train, header=header,
                                            tqdm_header='Batch',
                                            indent=indent)
        if change_train_eval:
            module.train()
        activate_params(module, params)




        
        for i, data in enumerate(loader_epoch):


            _iter = _epoch * len_loader_train + i
            match task:
                case 'emotion':

                    _input, _label, _soft_label, hapi_label  = data
                    
                    _input = _input.cuda()
                    if tea_model is not None:
                        m = nn.Softmax(dim=1)
                        with torch.no_grad():
                            _soft_label=m(tea_model(_input))
                        hapi_label = torch.argmax(_soft_label, dim=-1)
                    _soft_label = _soft_label.cuda()
                    _label = _label.cuda()
                    hapi_label = hapi_label.cuda()

                    _output = forward_fn(_input)

                    if adv_train and _epoch >7:
                      
                        pass
                    elif label_train:
                        loss = loss_fn( _label=_label, _output=_output)
                    elif hapi_label_train:
                        loss = loss_fn( _label=hapi_label, _output=_output)
                    else:
                        loss = loss_fn( _soft_label=_soft_label, _output=_output)

               
                case 'cifar10':
                    _input, _label  = data
                    _input = _input.cuda()
                    _label = _label.cuda()
                    hapi_label=_label
                                                               
                    if tea_model is not None:
                        m = nn.Softmax(dim=1)
                        with torch.no_grad():
                            _soft_label=m(tea_model(_input))
                        hapi_label = torch.argmax(_soft_label, dim=-1)
                    _output = forward_fn(_input)

                    if adv_train and _epoch >7:
                        pass
                    elif label_train:
                        loss = loss_fn( _label=_label, _output=_output)
                    elif hapi_label_train:
                        loss = loss_fn( _label=hapi_label, _output=_output)
                    else:
                        loss = loss_fn( _soft_label=_soft_label, _output=_output)
                    
            if backward_and_step and (adv_train == None or _epoch<8):
                optimizer.zero_grad()
                loss.backward()
                # if adv_train:
                #     _adv_soft_label, _adv_hapi_label = after_loss_fn(_input=_input,_soft_label=_soft_label,_output=_output,optimizer=optimizer)
                        
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

                #TODO 计算区别评估

            if lr_scheduler and lr_scheduler_freq == 'iter':
                lr_scheduler.step()
                
               
            if mixmatch:
                logger.update(n=batch_size, loss=float(loss))
            else:    
                match task:
                    case 'sentiment':
                        _output = _output[:,:2]
                        new_num_classes = 2
                    case 'emotion':
                        new_num_classes = num_classes
                    case 'cifar10':
                        new_num_classes = num_classes
                        
                hapi_acc1, hapi_acc5 = accuracy_fn(
                    _output, hapi_label, num_classes=new_num_classes, topk=(1, 5))
                gt_acc1, gt_acc5 = accuracy_fn(
                    _output, _label, num_classes=new_num_classes, topk=(1, 5))
                batch_size = int(_label.size(0)) 
                logger.update(n=batch_size, gt_acc1=gt_acc1,  
                            hapi_loss=float(loss), hapi_acc1=hapi_acc1)
        optimizer.zero_grad()
        
        
        
        if lr_scheduler and lr_scheduler_freq == 'epoch':
            lr_scheduler.step()
        if change_train_eval:
            module.eval()
        activate_params(module, [])
        if mixmatch:
            pass
        else:
            gt_acc1, hapi_loss, hapi_acc1 = (
                    logger.meters['gt_acc1'].global_avg,
                    logger.meters['hapi_loss'].global_avg,
                    logger.meters['hapi_acc1'].global_avg)
            if writer is not None:
                writer.add_scalars(main_tag='gt_acc1/' + main_tag,
                            tag_scalar_dict={tag: gt_acc1}, global_step=_epoch + start_epoch)        
                writer.add_scalars(main_tag='hapi_loss/' + main_tag,
                            tag_scalar_dict={tag: hapi_loss}, global_step=_epoch + start_epoch)
                writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
                        tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch + start_epoch)
        
        
                    
        if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
            validate_result = validate_fn(module=module,
                                          num_classes=num_classes,
                                          loader=loader_valid,
                                          writer=writer, tag=tag,
                                          _epoch=_epoch + start_epoch,
                                          verbose=verbose, indent=indent,
                                          label_train=label_train,
                                          hapi_label_train=hapi_label_train,
                                          api=api,task=task,after_loss_fn=after_loss_fn,adv_valid=adv_valid,tea_model=tea_model,
                                          **kwargs)
            cur_acc = validate_result[0]
            if cur_acc >= best_acc:
                best_validate_result = validate_result
                if verbose:
                    prints('{purple}best result update!{reset}'.format(
                        **ansi), indent=indent)
                    prints(f'Current Acc: {cur_acc:.3f}    '
                           f'Previous Best Acc: {best_acc:.3f}',
                           indent=indent)
                best_acc = cur_acc
                if save:
                    save_fn(log_dir=log_dir, module=module,verbose=verbose)
            if verbose:
                prints('-' * 50, indent=indent)
    module.zero_grad()
    return best_validate_result

