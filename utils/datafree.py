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
import torchvision
from adp_method.euclidean import euclidean_dist
from dataset.dataset import split_dataset


from dataset.kdef import KDEF


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

# !pip install pytorch_transformers

from dataset.imdb import IMDB
from dataset.rafdb import RAFDB
from torch.utils.data import Dataset, Subset

from torch.utils.tensorboard import SummaryWriter




@torch.no_grad()
def save_fn(log_dir, module, verbose: bool = False, indent: int = 0):
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


@torch.no_grad()
def missclassification_fn(_output: torch.Tensor, _label: torch.Tensor, hapi_label: torch.Tensor, num_classes: int,
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
    tt = 0  # hapi true and model true
    tf = 0  # hapi true and model false
    ft = 0  # hapi false and model true
    ff = 0  # hapi false and model false

    maxk = min(max(topk), num_classes)
    batch_size = _label.size(0)
    _, pred = _output.topk(maxk, 1, True, True)
    pred = pred.t()
    for i in range(batch_size):
        if hapi_label[i] == _label[i] and pred[0][i] == _label[i]:
            tt += 1
        elif hapi_label[i] == _label[i] and pred[0][i] != _label[i]:
            tf += 1
        elif hapi_label[i] != _label[i] and pred[0][i] == _label[i]:
            ft += 1
        elif hapi_label[i] != _label[i] and pred[0][i] != _label[i]:
            ff += 1

    res: list[float] = []

    res.append(tt * (100.0 / batch_size))
    res.append(tf * (100.0 / batch_size))
    res.append(ft * (100.0 / batch_size))
    res.append(ff * (100.0 / batch_size))
    return res


def val_loss(_input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, reduction: str = 'mean', **kwargs) -> torch.Tensor:

    criterion = nn.CrossEntropyLoss(reduction=reduction)

    return criterion(_output, _label)



def myprint(a):
    """Log the print statements"""
    global file
    print(a); file.write(a); file.write("\n"); file.flush()


def student_loss(_output, _soft_label, loss_type,return_soft_label=False):
    """Kl/ L1 Loss for student"""
    prin_soft_labels =  False
    if loss_type == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(_output, _soft_label.detach())
    elif loss_type == "kl":
        loss_fn = F.kl_div
        _output = F.log_softmax(_output, dim=1)
        _soft_label = F.softmax(_soft_label, dim=1)
        loss = loss_fn(_output, _soft_label.detach(), reduction="batchmean")

    if return_soft_label:
        return loss, _soft_label.detach()
    else:
        return loss

def generator_loss(args, _output, _soft_label,  z = None, z_logit = None, reduction="mean"):
    assert 0 
    
    loss = - F.l1_loss( _output, _soft_label , reduction=reduction) 
    
            
    return loss


def loss_fn(_input: torch.Tensor = None, _label: torch.Tensor = None,
            _output: torch.Tensor = None, reduction: str = 'mean', _soft_label: torch.Tensor = None,
            temp: float = 1.0, outputs_x=None, targets_x=None, outputs_u=None, targets_u=None, iter=None, total_iter=None,loss_type='l1',
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
        **kwargs: Keyword arguments passed to :meth:`ge_soft_labels()`
            if :attr:`_output` is not provided.

    Returns:
        torch.Tensor:
            A scalar loss tensor (with shape ``(N)`` if ``reduction='none'``).
    """
    if _soft_label is None:
        return val_loss(_input=_input, _label=_label, _output=_output, reduction=reduction)

    criterion = nn.CrossEntropyLoss(reduction=reduction)

    # return criterion(_output,_soft_label)+torch.mean((_output - _soft_label)**2)*100
    return student_loss(_output, _soft_label,loss_type)

from torchvision.utils import save_image


def get_api( x,  api='amazon', tea_model=None):
    adv_x_num = 500

 
    # convert the tensor to PIL image using above transform
    soft_label_batch = torch.zeros((x.shape[0], 8 if api =='amazon' else 7))
    hapi_label_batch = torch.zeros((x.shape[0]))
    update_x_batch = torch.zeros_like(x)

    noface_num = 0
    if tea_model is not None:
        m = nn.Softmax(dim=1)
        with torch.no_grad():
            soft_label_batch = m(tea_model(x))
        hapi_label_batch = torch.argmax(soft_label_batch, dim=-1)
        update_x_batch = x

        return update_x_batch, soft_label_batch, hapi_label_batch
    for i in range(x.shape[0]):
        # img:Image = transform(x[i,:,:,:])
        # img_input:Image = transform(_input[i,:,:,:])
        path = os.path.join('/data/jc/data/image/adv_x', str(adv_x_num) + '.png')
        save_image(x[i, :, :, :], path, 'png')

        adv_x_num += 1

        match api:
            case 'facepp':
                with io.open(path, 'rb') as image:
                    data = {'api_key': '_5FIJ5HcL3L5IQTfEEmAvRYbjL6QzGWb',
                            'api_secret': 'aEGVrumV8O0pACkQf-giP2R_3mcMTF9q',
                            'image_base64': base64.b64encode(image.read()),
                            'return_attributes': 'emotion'}
                    attempts = 0
                    success = False
                    while attempts < 3 and not success:
                        try:
                            r = requests.post(url='https://api-us.faceplusplus.com/facepp/v3/detect', data=data)
                            success = True
                        except:
                            attempts += 1
                            if attempts == 3:
                                print('no response')
                                os._exit(0)
                    responses = r.text
                    responses = json.loads(responses)
                    soft_label = torch.ones(7)

                    if len(responses['faces']) != 0:
                        soft_label[0] = responses['faces'][0]['attributes']['emotion']['anger']* 0.01
                        soft_label[1] = responses['faces'][0]['attributes']['emotion']['disgust']* 0.01
                        soft_label[2] = responses['faces'][0]['attributes']['emotion']['fear']* 0.01
                        soft_label[3] = responses['faces'][0]['attributes']['emotion']['happiness']* 0.01
                        soft_label[4] = responses['faces'][0]['attributes']['emotion']['sadness']* 0.01
                        soft_label[5] = responses['faces'][0]['attributes']['emotion']['surprise']* 0.01
                        soft_label[6] = responses['faces'][0]['attributes']['emotion']['neutral']* 0.01
                        hapi_label = torch.argmax(soft_label)
                        soft_label_batch[i - noface_num, :] = soft_label
                        hapi_label_batch[i - noface_num] = hapi_label
                        update_x_batch[i - noface_num, :, :, :] = x[i, :, :, :]
                       
                    else:
                        noface_num += 1

                        # print('no face')
                        # soft_label = torch.ones(7)*0.14285714285714285
                        # hapi_label = torch.tensor(6)

            case 'amazon':

                with io.open(path, 'rb') as image:
                    responses = client.detect_faces(Image={'Bytes': image.read()}, Attributes=["ALL"])
                    soft_label = torch.ones(8)

                    if len(responses['FaceDetails']) != 0:
                        api_result = [{
                            responses['FaceDetails'][0]['Emotions'][0]["Type"]: responses['FaceDetails'][0]['Emotions'][0]["Confidence"],
                            responses['FaceDetails'][0]['Emotions'][1]["Type"]: responses['FaceDetails'][0]['Emotions'][1]["Confidence"],
                            responses['FaceDetails'][0]['Emotions'][2]["Type"]: responses['FaceDetails'][0]['Emotions'][2]["Confidence"],
                            responses['FaceDetails'][0]['Emotions'][3]["Type"]: responses['FaceDetails'][0]['Emotions'][3]["Confidence"],
                            responses['FaceDetails'][0]['Emotions'][4]["Type"]: responses['FaceDetails'][0]['Emotions'][4]["Confidence"],
                            responses['FaceDetails'][0]['Emotions'][5]["Type"]: responses['FaceDetails'][0]['Emotions'][5]["Confidence"],
                            responses['FaceDetails'][0]['Emotions'][6]["Type"]: responses['FaceDetails'][0]['Emotions'][6]["Confidence"],
                            responses['FaceDetails'][0]['Emotions'][7]["Type"]: responses['FaceDetails'][0]['Emotions'][7]["Confidence"]

                        }]
                        soft_label[0] = api_result[0]['ANGRY'] * 0.01
                        soft_label[1] = api_result[0]['DISGUSTED'] * 0.01
                        soft_label[2] = api_result[0]['FEAR'] * 0.01
                        soft_label[3] = api_result[0]['HAPPY'] * 0.01
                        soft_label[4] = api_result[0]['SAD'] * 0.01
                        soft_label[5] = api_result[0]['SURPRISED'] * 0.01
                        soft_label[6] = api_result[0]['CALM'] * 0.01 
                        soft_label[7] = api_result[0]['CONFUSED'] * 0.01
                        hapi_label = torch.argmax(soft_label)
                        soft_label_batch[i - noface_num, :] = soft_label
                        hapi_label_batch[i - noface_num] = hapi_label
                        update_x_batch[i - noface_num, :, :, :] = x[i, :, :, :]
                     
                    else:
                        # 'HAPPY'|'SAD'|'ANGRY'|'CONFUSED'|'DISGUSTED'|'SURPRISED'|'CALM'|'UNKNOWN'|'FEAR',
                        # print('no face')
                        # soft_label = torch.ones(7)*0.14285714285714285
                        # hapi_label = torch.tensor(6)
                        noface_num += 1

            case _:
                raise NotImplementedError

    return update_x_batch[:x.shape[0] - noface_num], soft_label_batch[:x.shape[0] - noface_num], hapi_label_batch[:x.shape[0] - noface_num]



def estimate_gradient_objective(G_activation,loss_type,no_logits,logit_correction,api,  clone_model, x, forward_differences,epsilon = 1e-7, m = 5, verb=False, num_classes=10, device = "cuda", pre_x=False):
    # Sampling from unit sphere is the method 3 from this website:
    #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    #x = torch.Tensor(np.arange(2*1*7*7).reshape(-1, 1, 7, 7))
    
    if pre_x and G_activation is None:
        raise ValueError(G_activation)


    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)
        C = x.size(1)
        S = x.size(2)
        dim = S**2 * C

        u = np.random.randn(N * m * dim).reshape(-1, m, dim) # generate random points from normal distribution

        d = np.sqrt(np.sum(u ** 2, axis = 2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim = 1) # Shape N, m + 1, S^2

            

        u = u.view(-1, m + 1, C, S, S)

        evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
        if pre_x: 
            evaluation_points = G_activation(evaluation_points) # Apply G_activation function

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_clone = []
        max_number_points = 32*156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU
        
        for i in (range(N * m // max_number_points + 1)): 
            pts = evaluation_points[i * max_number_points: (i+1) * max_number_points]
            pts = pts.to(device)
            
            with torch.no_grad():
                pts,pred_victim_pts,_ = get_api(pts,api)
                pts = pts.detach()
                pred_victim_pts = pred_victim_pts.detach()
                
                pred_clone_pts = clone_model(pts)

            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)



        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_clone = torch.cat(pred_clone, dim=0).to(device)

        u = u.to(device)

        if loss_type == "l1":
            loss_fn = F.l1_loss
            if no_logits:
                pred_victim = F.log_softmax(pred_victim, dim=1).detach()
                if logit_correction == 'min':
                    pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()
                elif logit_correction == 'mean':
                    pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()


        elif loss_type == "kl":
            loss_fn = F.kl_div
            pred_clone = F.log_softmax(pred_clone, dim=1)
            pred_victim = F.softmax(pred_victim.detach(), dim=1)


        # Compute loss
        if loss_type == "kl":
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').sum(dim = 1).view(-1, m + 1) 
        else:
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').mean(dim = 1).view(-1, m + 1) 

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        if forward_differences:
            gradient_estimates *= dim            

        if loss_type == "kl":
            gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, S, S) 
        else:
            gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, S, S) / (num_classes * N) 

        
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


def compute_gradient(api,  clone_model, x, G_activation,loss_type,no_logits,logit_correction,pre_x=False, device="cuda"):
    if pre_x and G_activation is None:
        raise ValueError(G_activation)

    N = x.size(0)
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)


    if pre_x:
        x_ = G_activation(x_)


    pred_victim = get_api(api,x_)
    
    with torch.no_grad():
        pred_clone = clone_model(x_)

    if loss_type == "l1":
        loss_fn = F.l1_loss
        if no_logits:
            pred_victim_no_logits = F.log_softmax(pred_victim, dim=1)
            if logit_correction == 'min':
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.min(dim=1).values.view(-1, 1)
            elif logit_correction == 'mean':
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.mean(dim=1).view(-1, 1)
            else:
                pred_victim = pred_victim_no_logits

    elif loss_type == "kl":
        loss_fn = F.kl_div
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_victim = F.softmax(pred_victim, dim=1)

    else:
        raise ValueError(loss_type)


    loss_values = -loss_fn(pred_clone, pred_victim, reduction='mean')
    # print("True mean loss", loss_values)
    loss_values.backward()

    clone_model.train()
    
    return x_copy.grad, loss_values

def measure_true_grad_norm(api, forward_fn, x,G_activation,loss_type,no_logits,logit_correction):
    # Compute true gradient of loss wrt x
    
    true_grad, _ = compute_gradient(api, forward_fn, x, G_activation,loss_type,no_logits,logit_correction,pre_x=True)
    true_grad = true_grad.view(-1, 3072)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()

    return norm_grad



def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            S_grad.append(p.grad.norm().to("cpu"))
    return  np.mean(G_grad), np.mean(S_grad)

def distillation(module: nn.Module,  num_classes: int,
                 epochs: int,  lr_scheduler_S, lr_scheduler_G,
                 log_dir: str = 'runs/test',
                 grad_clip: float = 5.0,
                 print_prefix: str = 'Distill', start_epoch: int = 0, resume: int = 0,
                 
                 validate_interval: int = 1, save: bool = True,
                 
                 loader_valid: torch.utils.data.DataLoader = None,
                 main_tag: str = 'train', tag: str = '',
                 verbose: bool = True, output_freq: str = 'iter', indent: int = 0,
                 change_train_eval: bool = True, lr_scheduler_freq: str = 'epoch',
                  label_train: bool = False, hapi_label_train: bool = False,
                 api=False, task='sentiment', 
                 batch_size=None, 
                  tea_model=None,
                 generator=None,grad_epsilon=None,grad_m=None,loss_type=None,
                    epoch_itrs=None,nz=None,optimizer_G=None,optimizer_S=None,
                    g_iter=None,d_iter=None,approx_grad=None,rec_grad_norm=None,logit_correction=None,no_logits=None,
                 G_activation=None,forward_differences=None,
                 **kwargs):
    r"""Train the model"""


    if epochs <= 0:
        return
    forward_fn = module.__call__

    writer = SummaryWriter(log_dir=log_dir)
    validate_fn = dis_validate

    scaler: torch.cuda.amp.GradScaler = None

    best_validate_result = (0.0, float('inf'))
    best_acc = 0.0
    best_loss = 100.0
    # if validate_interval != 0:
    #     validate_result = validate_fn(module=module,
    #                                 num_classes=num_classes,
    #                                 loader=loader_valid,
    #                                 writer=writer, tag=tag,
    #                                 _epoch=start_epoch,
    #                                 verbose=verbose, indent=indent,
    #                                 label_train=label_train,
    #                                 api=api,task=task,after_loss_fn=after_loss_fn,adv_valid=adv_valid,
    #                                 **kwargs)
    #     best_acc = best_validate_result[0]

    params_S: list[nn.Parameter] = []
    params_G: list[nn.Parameter] = []
    for param_group in optimizer_S.param_groups:
        params_S.extend(param_group['params'])
    for param_group in optimizer_G.param_groups:
        params_G.extend(param_group['params'])

    logger = MetricLogger()

    logger.create_meters(loss_G=None,
                             loss_S=None)
    if resume and lr_scheduler_S:
        for _ in range(resume):
            lr_scheduler_S.step()
            lr_scheduler_G.step()
    iterator = range(resume, epochs)
    if verbose and output_freq == 'epoch':
        header: str = '{blue_light}{0}: {reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(header), 30) + get_ansi_len(header))
        iterator = logger.log_every(range(resume, epochs),
                                    header=print_prefix,
                                    tqdm_header='Epoch',
                                    indent=indent)

    for _epoch in iterator:
        norm_par = module.norm_par
        module.norm_par = None
    
        _epoch += 1
        logger.reset()
    
        if verbose and output_freq == 'iter':
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            loader_epoch = logger.log_every(range(epoch_itrs), header=header,
                                            tqdm_header='Iter',
                                            indent=indent)
        if change_train_eval:
            module.train()
        activate_params(module, params_S)
        activate_params(module, params_G)
        #TODO:optimizer


        for i, _ in enumerate(loader_epoch):
            for _ in range(g_iter):
                #Sample Random Noise
                z = torch.randn((batch_size, nz)).cuda()
                optimizer_G.zero_grad()
                generator.train()
                
                #Get fake image from generator
                fake = generator(z, pre_x=approx_grad) # pre_x returns the output of G before applying the activation


                ## APPOX GRADIENT
                approx_grad_wrt_x, loss_G = estimate_gradient_objective( G_activation,loss_type,no_logits,logit_correction,
                                                                        api,forward_fn, fake,forward_differences,
                                                    epsilon = grad_epsilon, m = grad_m, num_classes=num_classes, pre_x=True)

                fake.backward(approx_grad_wrt_x)
                    
                optimizer_G.step()

                if i == 0 and rec_grad_norm:
                    x_true_grad = measure_true_grad_norm(api, forward_fn,  fake,G_activation,loss_type,no_logits,logit_correction)

                    
            for _ in range(d_iter):
                z = torch.randn((batch_size, nz)).cuda()
                fake = generator(z).detach()
                optimizer_S.zero_grad()

                with torch.no_grad(): 
                    t_logit = get_api(fake,api)

                # Correction for the fake logits
                if loss_type == "l1" and no_logits:
                    t_logit = F.log_softmax(t_logit, dim=1).detach()
                    if logit_correction == 'min':
                        t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                    elif logit_correction == 'mean':
                        t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()


                s_logit = forward_fn(fake)


                loss_S = student_loss( s_logit, t_logit,loss_type)
                loss_S.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params_S, grad_clip)
                optimizer_S.step()
                
                if lr_scheduler_S and lr_scheduler_freq == 'iter':
                    lr_scheduler_S.step()
                    lr_scheduler_G.step()

                logger.update(n=batch_size, loss_G=loss_G.item(),
                                loss_S=loss_S.item())
                
                if rec_grad_norm and i == 0:
                    G_grad_norm, S_grad_norm = compute_grad_norms(generator, forward_fn)
                    with open(log_dir + "/norm_grad.csv", "a") as f:
                        f.write("%d,%f,%f,%f\n"%(_epoch + start_epoch, G_grad_norm, S_grad_norm, x_true_grad))
            # end of iter

            optimizer_S.zero_grad()
            optimizer_G.zero_grad()

        # end of epoch
        if lr_scheduler_S and lr_scheduler_freq == 'epoch':
            lr_scheduler_S.step()
            lr_scheduler_G.step()
            
        if change_train_eval:
            module.eval()
        activate_params(module, [])
        activate_params(generator, [])

        loss_G, loss_S = (
            logger.meters['loss_G'].global_avg,
            logger.meters['loss_S'].global_avg
            )
        if writer is not None:
            writer.add_scalars(main_tag='loss_G/' + main_tag,
                                tag_scalar_dict={tag: loss_G}, global_step=_epoch + start_epoch)
            writer.add_scalars(main_tag='loss_S/' + main_tag,
                                tag_scalar_dict={tag: loss_S}, global_step=_epoch + start_epoch)
         
            with open(log_dir + "/loss.csv", "a") as f:
                f.write("%d,%f,%f\n"%(_epoch + start_epoch, loss_G, loss_S))
                
       
        if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
            module.norm_par = norm_par
            
            validate_result = validate_fn(module=module,
                                          num_classes=num_classes,
                                          loader=loader_valid,
                                          writer=writer, tag=tag,
                                          _epoch=_epoch + start_epoch,
                                          verbose=verbose, indent=indent,
                                          label_train=label_train,
                                          hapi_label_train=hapi_label_train, 
                                          task=task, tea_model=tea_model,
                                          **kwargs)
            
            if label_train:
                cur_acc = validate_result[2]
            else:    
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
                    save_fn(log_dir=log_dir, module=module, verbose=verbose)
            if verbose:
                prints('-' * 50, indent=indent)
    module.zero_grad()
    print('best_validate_result', best_validate_result)
    return best_validate_result


def dis_validate(module: nn.Module, num_classes: int,
                 loader: torch.utils.data.DataLoader,
                 print_prefix: str = 'Validate', indent: int = 0,
                 verbose: bool = True,
                 writer=None, main_tag: str = 'valid',
                 tag: str = '', _epoch: int = None,
                 label_train=False, hapi_label_train=False, task=None, 
                 tea_model=None,
             
                 **kwargs) -> tuple[float, float]:
    r"""Evaluate the model.

    Returns:
        (float, float): Accuracy and loss.
    """
    module.eval()

    forward_fn = module.__call__

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

    logger.create_meters(gt_loss=None, gt_acc1=None,
                             hapi_loss=None, hapi_acc1=None)

    loader_epoch = loader
    if verbose:
        header: str = '{yellow}{0}{reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))
        loader_epoch = logger.log_every(loader, header=header,
                                        tqdm_header='Epoch',
                                        indent=indent)

    for data in loader_epoch:

        with torch.no_grad():
            match task:
                case 'emotion':

                    _input, _label, _soft_label, hapi_label = data
                    _input = _input.cuda()
                    if tea_model is not None:
                        m = nn.Softmax(dim=1)
                        with torch.no_grad():
                            _soft_label = m(tea_model(_input))
                        hapi_label = torch.argmax(_soft_label, dim=-1)
                    _soft_label = _soft_label.cuda()
                    _label = _label.cuda()
                    hapi_label = hapi_label.cuda()

                 
                    _output = forward_fn(_input)

                case 'sentiment':
                    if module.model_name == 'xlnet':
                        input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label = data
                        token_type_ids = token_type_ids.cuda()
                    elif module.model_name == 'roberta' or  module.model_name == 't5':
                        input_ids, attention_mask, _label, _soft_label, hapi_label = data
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    _label = _label.cuda()
                    _soft_label = _soft_label.cuda()
                    hapi_label = hapi_label.cuda()
                    if module.model_name == 'xlnet':
                        _output = forward_fn(input_ids=input_ids, token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
                    elif module.model_name == 'roberta' or  module.model_name == 't5':
                        _output = forward_fn(input_ids=input_ids, attention_mask=attention_mask)
                  
                case 'cifar10':
                    _input, _label = data
                    _input = _input.cuda()
                    _label = _label.cuda()
                    hapi_label = _label
                    if tea_model is not None:
                        m = nn.Softmax(dim=1)
                        with torch.no_grad():
                            _soft_label = m(tea_model(_input))
                        hapi_label = torch.argmax(_soft_label, dim=-1)
                    _output = forward_fn(_input)


            gt_loss = float(loss_fn(_label=_label, _output=_output, **kwargs))
            if label_train:
                hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))
            elif hapi_label_train:
                hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))
            else:
                hapi_loss = float(loss_fn(_soft_label=_soft_label, _output=_output, **kwargs))

            batch_size = int(_label.size(0))
            
            hapi_acc1, hapi_acc5 = accuracy_fn(
                _output, hapi_label, num_classes=num_classes, topk=(1, 5))
            match task:
                case 'sentiment':
                    _output = _output[:, :2]
                    new_num_classes = 2
                    
                case 'emotion':
                    new_num_classes = num_classes
                case 'cifar10':
                    new_num_classes = num_classes
            gt_acc1, gt_acc5 = accuracy_fn(
                _output, _label, num_classes=new_num_classes, topk=(1, 5))
           
            logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                                hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1)

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


    return hapi_acc1, hapi_loss, gt_acc1

