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
from dataset.advdataset import advdataset
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


def entropy(y_pred_prob, indices, n_samples):

    origin_index = torch.tensor(indices).cuda()
    entropy = -torch.nansum(torch.multiply(y_pred_prob, torch.log(y_pred_prob)), axis=1)
    pred_label = torch.argmax(y_pred_prob, axis=1)
    eni = torch.column_stack((origin_index[:len(entropy)],
                              entropy,
                              pred_label))

    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:, 0].type(torch.IntTensor)[:n_samples]


def mixmatch_get_data(data, forward_fn, unlabel_iterator):
    # https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py
    r"""Process data. Defaults to be :attr:`self.dataset.get_data`.
    If :attr:`self.dataset` is ``None``, return :attr:`data` directly.

    Args:
        data (Any): Unprocessed data.
        **kwargs: Keyword arguments passed to
            :attr:`self.dataset.get_data()`.

    Returns:
        Any: Processed data.
    """
    # TODO:MODE

    T: float = 0.5
    alpha: float = 0.75
    mixmatch: bool = False
    lambda_u: float = 100.0
    _input, _label, _soft_label, hapi_label = data
    _input = _input.cuda()
    _soft_label = _soft_label.cuda()
    _label = _label.cuda()
    hapi_label = hapi_label.cuda()
    (inputs_u, inputs_u2), _, _, _ = next(unlabel_iterator)
    inputs_u = inputs_u.cuda()
    inputs_u2 = inputs_u2.cuda()

    with torch.no_grad():
        # compute guessed labels of unlabel samples
        outputs_u = forward_fn(inputs_u)
        outputs_u2 = forward_fn(inputs_u2)
        p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        pt = p**(1 / T)
        targets_u = pt / pt.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()

    # mixup
    all_inputs = torch.cat([_input, inputs_u, inputs_u2], dim=0)
    all_targets = torch.cat([_soft_label, targets_u, targets_u], dim=0)

    l = np.random.beta(alpha, alpha)

    l = max(l, 1 - l)

    idx = torch.randperm(all_inputs.size(0))

    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target, _input.shape[0]


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


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave_fn(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


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

# @torch.no_grad()
# def amazon_accuracy_fn(_output: torch.Tensor, _label: torch.Tensor, num_classes: int,
#              topk: Iterable[int] = (1, 5)) -> list[float]:
#     r"""Computes the accuracy over the k top predictions
#     for the specified values of k.

#     Args:
#         _output (torch.Tensor): The batched logit tensor with shape ``(N, C)``.
#         _label (torch.Tensor): The batched label tensor with shape ``(N)``.
#         num_classes (int): Number of classes.
#         topk (~collections.abc.Iterable[int]): Which top-k accuracies to show.
#             Defaults to ``(1, 5)``.

#     Returns:
#         list[float]: Top-k accuracies.
#     """
#     maxk = min(max(topk), num_classes)
#     batch_size = _label.size(0)
#     _output = _output[:,:2]
#     _, pred = _output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(_label[None])
#     res: list[float] = []
#     for k in topk:
#         if k > num_classes:
#             res.append(100.0)
#         else:
#             correct_k = float(correct[:k].sum(dtype=torch.float32))
#             res.append(correct_k * (100.0 / batch_size))
#     return res


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


def linear_rampup(iter, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(iter / rampup_length, 0.0, 1.0)
        return float(current)


def SemiLoss(outputs_x, targets_x, outputs_u, targets_u, iter, total_iter):
    # Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/train_iteration)
    # lx is cross entropy, lu is L2 normalization

    probs_u = torch.softmax(outputs_u, dim=1)
    Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    Lu = torch.mean((probs_u - targets_u)**2)
    lambda_u = 100.0
    w = lambda_u * linear_rampup(iter, total_iter)
    loss = Lx + w * Lu
    return loss


def loss_fn(_input: torch.Tensor = None, _label: torch.Tensor = None,
            _output: torch.Tensor = None, reduction: str = 'mean', _soft_label: torch.Tensor = None,
            temp: float = 1.0, outputs_x=None, targets_x=None, outputs_u=None, targets_u=None, iter=None, total_iter=None,
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
        return SemiLoss(outputs_x, targets_x, outputs_u, targets_u, iter, total_iter)

    if _soft_label is None:
        return val_loss(_input=_input, _label=_label, _output=_output, reduction=reduction)

    criterion = nn.CrossEntropyLoss(reduction=reduction)

    # return criterion(_output,_soft_label)+torch.mean((_output - _soft_label)**2)*100
    return criterion(_output, _soft_label)


# def entropy(y_pred_prob, n_samples):
#     origin_index = np.arange(0, len(y_pred_prob))
#     entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
#     pred_label = np.argmax(y_pred_prob, axis=1)
#     eni = np.column_stack((origin_index,
#                            entropy,
#                            pred_label))

#     eni = eni[(-eni[:, 1]).argsort()]
#     return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]


from torchvision.utils import save_image



def get_api(_input, x, indices, api='amazon', tea_model=None):
    adv_x_num = 500

    # define a transform to convert a tensor to PIL image
    transform = T.ToPILImage(mode='RGB')
    convert_tensor = T.ToTensor()

    # convert the tensor to PIL image using above transform
    soft_label_batch = torch.zeros((x.shape[0], 8 if api =='amazon' else 7))
    hapi_label_batch = torch.zeros((x.shape[0]))
    adv_x_batch = torch.zeros_like(x)
    new_indices = torch.zeros((x.shape[0]), dtype=torch.long)
    noface_num = 0
    if tea_model is not None:
        m = nn.Softmax(dim=1)
        with torch.no_grad():
            soft_label_batch = m(tea_model(x))
        hapi_label_batch = torch.argmax(soft_label_batch, dim=-1)
        adv_x_batch = x
        new_indices = indices
        return adv_x_batch, soft_label_batch, hapi_label_batch, new_indices
    for i in range(x.shape[0]):
        # img:Image = transform(x[i,:,:,:])
        # img_input:Image = transform(_input[i,:,:,:])
        path = os.path.join('/data/jc/data/image/adv_x', str(adv_x_num) + '.png')
        path_input = os.path.join('/data/jc/data/image/adv_x', str(adv_x_num) + '_input.png')
        save_image(x[i, :, :, :], path, 'png')
        save_image(_input[i, :, :, :], path_input, 'png')
        # path_b = os.path.join('/data/jc/data/image/adv_x', str(adv_x_num)+'b'+'.png')
        # path_input_b = os.path.join('/data/jc/data/image/adv_x', str(adv_x_num)+'b'+'_input.png')
        # save_image_b(x[i,:,:,:],path_b)
        # save_image_b(_input[i,:,:,:],path_input_b)
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
                        adv_x_batch[i - noface_num, :, :, :] = x[i, :, :, :]
                        new_indices[i - noface_num] = indices[i]
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
                        adv_x_batch[i - noface_num, :, :, :] = x[i, :, :, :]
                        new_indices[i - noface_num] = indices[i]

                    else:
                        # 'HAPPY'|'SAD'|'ANGRY'|'CONFUSED'|'DISGUSTED'|'SURPRISED'|'CALM'|'UNKNOWN'|'FEAR',
                        # print('no face')
                        # soft_label = torch.ones(7)*0.14285714285714285
                        # hapi_label = torch.tensor(6)
                        noface_num += 1

            case _:
                raise NotImplementedError

    return adv_x_batch[:x.shape[0] - noface_num], soft_label_batch[:x.shape[0] - noface_num], hapi_label_batch[:x.shape[0] - noface_num], new_indices[:x.shape[0] - noface_num]


def distillation(module: nn.Module, pgd_set, num_classes: int,
                 epochs: int, optimizer, lr_scheduler, adv_train=None, adv_train_iter=7, adv_valid=False,
                 log_dir: str = 'runs/test',
                 grad_clip: float = 5.0,
                 print_prefix: str = 'Distill', start_epoch: int = 0, resume: int = 0,
                 validate_interval: int = 1, save: bool = True,
                 loader_train: torch.utils.data.DataLoader = None,
                 loader_valid: torch.utils.data.DataLoader = None,
                 unlabel_iterator=None,
                 file_path: str = None,
                 folder_path: str = None, suffix: str = None,
                 main_tag: str = 'train', tag: str = '',

                 verbose: bool = True, output_freq: str = 'iter', indent: int = 0,
                 change_train_eval: bool = True, lr_scheduler_freq: str = 'epoch',
                 backward_and_step: bool = True,
                 mixmatch: bool = False, label_train: bool = False, hapi_label_train: bool = False,
                 api=False, task='sentiment', unlabel_dataset_indices=None,
                 hapi_data_dir=None, hapi_info=None,
                 batch_size=None, num_workers=None,
                 n_samples=None, adaptive=None, get_sampler_fn=None,
                 balance=False, sample_times=10, tea_model=None, AE=None, encoder_attack=False,
                 pgd_percent=None,
                 encoder_train=False,train_dataset=None,
                workers=8,
                 **kwargs):
    r"""Train the model"""
    start_pgd_epoch = 1
    end_pgd_epoch = 30
    
    if epochs <= 0:
        return
    after_loss_fn = None
    forward_fn = module.__call__
    already_selected = []
    
    if adaptive == 'cloudleak':
        from trojanzoo.optim import PGD
        pgd = PGD(pgd_alpha=2 / 255, pgd_eps=4 / 255, iteration=7, random_init=True)

        def after_loss_fn(_input: torch.Tensor, _output: torch.Tensor = None
                          ):


            model_label = torch.argmax(_output, dim=-1)

            def pgd_loss_fn(_input: torch.Tensor,target:torch.Tensor,**kwargs):
                if encoder_attack:
                    return -F.cross_entropy(forward_fn(AE.decoder(_input)), target)
                return -F.cross_entropy(forward_fn(_input), target)

            @torch.no_grad()
            def early_stop_check(current_idx: torch.Tensor,
                                    adv_input: torch.Tensor, target: torch.Tensor, *args,
                                    stop_threshold: float = None, require_class: bool = None,
                                    **kwargs) -> torch.Tensor:
                if encoder_attack:
                    _class = torch.argmax(forward_fn(AE.decoder(adv_input[current_idx])), dim=-1)
                else:
                    _class = torch.argmax(forward_fn(adv_input[current_idx]), dim=-1)
                class_result = _class == target[current_idx]
                class_result = ~class_result
                result = class_result
                return result.detach().cpu()
            pgd.early_stop_check = early_stop_check

            adv_x, succ_tensor = pgd.optimize(_input=_input, loss_fn=pgd_loss_fn,target=model_label,loss_kwargs={'target':model_label})
            
            succ_tensor = succ_tensor.eq(-1)
            adv_x, _adv_soft_label, _adv_hapi_label, new_indices = get_api(
                _input, adv_x, np.arange(adv_x.shape[0]), api)
           
            return adv_x, _adv_soft_label, _adv_hapi_label,new_indices

    writer = SummaryWriter(log_dir=log_dir)
    validate_fn = dis_validate

    scaler: torch.cuda.amp.GradScaler = None

    best_validate_result = (0.0, float('inf'))
    best_acc = 0.0
    best_loss = 100.0
    
    params: list[nn.Parameter] = []
    for param_group in optimizer.param_groups:
        params.extend(param_group['params'])


    logger = MetricLogger()
    if mixmatch or encoder_train:
        logger.create_meters(loss=None)
    elif adv_train:
        logger.create_meters(gt_acc1=None,
                             hapi_loss=None, hapi_acc1=None, attack_succ=None, ahapi_succ=None)
    else:
        logger.create_meters(gt_acc1=None,
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
      
        if adaptive == 'cloudleak':
            if sample_times != 0 and (_epoch+1) %2 ==0:
                loader_train = None
                sample_times -= 1
                new_index = np.arange(len(train_dataset))[~np.in1d(np.arange(len(train_dataset)),already_selected)]
                selection_result = np.random.choice(new_index,n_samples,replace=False)
                already_selected.extend(selection_result)
                print("new_index",new_index.shape,"  already_selected",len(already_selected))
                #the clean input need to query in this epoch
                dst_subset = torch.utils.data.Subset(train_dataset, selection_result)
                loader_dst = torch.utils.data.DataLoader(dst_subset, batch_size=batch_size, shuffle=False,
                                                        num_workers=workers, pin_memory=True,drop_last=False)

                for i,data in enumerate(loader_dst):
                    # input_batch,_,_,_ = data
                    input_batch, _label, _soft_label, hapi_label= data
                    # input_batch = input_batch.cuda()
                    # with torch.no_grad():
                    #     _output=forward_fn(input_batch)
                    # # generate adv_x and query
                    # adv_x, _adv_soft_label, _adv_hapi_label,new_indices = after_loss_fn(_input=input_batch, _output=_output)
                    if i == 0 and _epoch == 1:
                        Synthetic_x = input_batch
                        Synthetic_adv_x = input_batch
                        Synthetic_adv_soft_label = _soft_label
                        Synthetic_adv_hapi_label = hapi_label
                    # input_batch = input_batch.cuda()
                    else:
                        Synthetic_x = torch.cat((Synthetic_x,input_batch),dim=0) 
                        Synthetic_adv_x = torch.cat((Synthetic_adv_x,input_batch),dim=0) 
                        Synthetic_adv_soft_label = torch.cat((Synthetic_adv_soft_label,_soft_label),dim=0)  
                        Synthetic_adv_hapi_label = torch.cat((Synthetic_adv_hapi_label,hapi_label),dim=0)  
                adv_dataset = advdataset(Synthetic_x.cpu(),Synthetic_adv_x.cpu(), Synthetic_adv_soft_label.cpu(),Synthetic_adv_hapi_label.cpu())
                loader_train = torch.utils.data.DataLoader(adv_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=workers, pin_memory=True,drop_last=False)
                
               
            
            
            
        loader_epoch = loader_train
        len_loader_train = len(loader_train)
        total_iter = (epochs - resume) * len_loader_train
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

            _, _input, _soft_label, hapi_label = data
            ori_img = None
            ori_soft = None
            _input = _input.cuda()
            _soft_label = _soft_label.cuda()
            hapi_label = hapi_label.cuda()
            _output = forward_fn(_input)
            if hapi_label_train:
                loss = loss_fn(_label=hapi_label, _output=_output)
            else:
                loss = loss_fn(_soft_label=_soft_label, _output=_output)    
            if backward_and_step:
                optimizer.zero_grad()
                loss.backward()

                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

           

            if lr_scheduler and lr_scheduler_freq == 'iter':
                lr_scheduler.step()


                hapi_acc1, hapi_acc5 = accuracy_fn(
                    _output, hapi_label, num_classes=num_classes, topk=(1, 5))
                batch_size = int(hapi_label.size(0))
                logger.update(n=batch_size, hapi_loss=float(loss), hapi_acc1=hapi_acc1)
        optimizer.zero_grad()

        if lr_scheduler and lr_scheduler_freq == 'epoch':
            lr_scheduler.step()
        if change_train_eval:
            module.eval()
        activate_params(module, [])

        hapi_loss, hapi_acc1 = (
            logger.meters['hapi_loss'].global_avg,
            logger.meters['hapi_acc1'].global_avg)
        if writer is not None:
            
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
                                          hapi_label_train=hapi_label_train, encoder_train=encoder_train,
                                          api=api, task=task, after_loss_fn=after_loss_fn, adv_valid=adv_valid, tea_model=tea_model,
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
                 label_train=False, hapi_label_train=False, api=False, task=None, after_loss_fn=None, adv_valid=False, tea_model=None,
                 encoder_train=False,
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
                                        tqdm_header='Epoch',
                                        indent=indent)
    encoder_num = 0
    for data in loader_epoch:
        if adv_valid:
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
                    if adv_valid:
                        loss, adv_x, adv_api_soft_label, adv_api_hapi_label = after_loss_fn(
                            _input=_input, _label=_label, _output=_output, mode='valid')
                        adv_x = adv_x.cuda()
                        adv_api_soft_label = adv_api_soft_label.cuda()
                        adv_output = forward_fn(adv_x)
                        adv_loss = float(loss_fn(_soft_label=adv_api_soft_label, _output=adv_output, **kwargs))

                case 'sentiment':
                    if module.model_name == 'xlnet':
                        input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label = data
                        token_type_ids = token_type_ids.cuda()
                    elif module.model_name == 'roberta':
                        input_ids, attention_mask, _label, _soft_label, hapi_label = data
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    _label = _label.cuda()
                    _soft_label = _soft_label.cuda()
                    hapi_label = hapi_label.cuda()
                    if module.model_name == 'xlnet':
                        _output = forward_fn(input_ids=input_ids, token_type_ids=token_type_ids,
                                             attention_mask=attention_mask)
                    elif module.model_name == 'roberta':
                        _output = forward_fn(input_ids=input_ids, attention_mask=attention_mask)

                    # # input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label  = data
                    # input_ids, attention_mask, _label, _soft_label, hapi_label  = data
                    # input_ids = input_ids.cuda()
                    # # token_type_ids = token_type_ids.cuda()
                    # attention_mask = attention_mask.cuda()
                    # _label = _label.cuda()
                    # _soft_label = _soft_label.cuda()
                    # hapi_label = hapi_label.cuda()

                    # _output = forward_fn(input_ids=input_ids,attention_mask=attention_mask)
                    if adv_valid:
                        raise NotImplementedError(f'{adv_valid=} is not supported on sentiment yet.')

            gt_loss = float(loss_fn(_label=_label, _output=_output, **kwargs))
            if label_train:
                hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))
            elif hapi_label_train:
                hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))
            else:
                hapi_loss = float(loss_fn(_soft_label=_soft_label, _output=_output, **kwargs))

            batch_size = int(_label.size(0))
            match task:
                case 'sentiment':
                    _output = _output[:, :2]
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

                logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                              hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1,
                              adv_loss=adv_loss, adv_acc1=adv_acc1)
            else:
                logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                              hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1)
        else:
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

                        if encoder_train:
                            encoded, _output = forward_fn(_input)
                        else:
                            _output = forward_fn(_input)

                    case 'sentiment':
                        if module.model_name == 'xlnet':
                            input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label = data
                            token_type_ids = token_type_ids.cuda()
                        elif module.model_name == 'roberta':
                            input_ids, attention_mask, _label, _soft_label, hapi_label = data
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        _label = _label.cuda()
                        _soft_label = _soft_label.cuda()
                        hapi_label = hapi_label.cuda()
                        if module.model_name == 'xlnet':
                            _output = forward_fn(input_ids=input_ids, token_type_ids=token_type_ids,
                                                 attention_mask=attention_mask)
                        elif module.model_name == 'roberta':
                            _output = forward_fn(input_ids=input_ids, attention_mask=attention_mask)
                        if adv_valid:
                            raise NotImplementedError(f'{adv_valid=} is not supported on sentiment yet.')
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
                        if encoder_train:
                            encoded, _output = forward_fn(_input)
                        else:
                            _output = forward_fn(_input)

                if encoder_train:
                    pass
                else:
                    gt_loss = float(loss_fn(_label=_label, _output=_output, **kwargs))
                if label_train:
                    hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))
                elif hapi_label_train:
                    hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))
                elif encoder_train:
                    criterion = nn.BCELoss()
                    loss = criterion(_output, _input)
                    logger.update(n=_input.size(0), loss=float(loss))
                    x = torchvision.utils.make_grid(torch.cat((_input, _output), dim=0))
                    path = os.path.join('/data/jc/data/image/encodelion', str(encoder_num) + '.png')
                    encoder_num += 1
                    save_image(x, path, 'png')

                    continue
                else:
                    hapi_loss = float(loss_fn(_soft_label=_soft_label, _output=_output, **kwargs))

                batch_size = int(_label.size(0))
                match task:
                    case 'sentiment':
                        _output = _output[:, :2]
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

                    logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                                  hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1,
                                  adv_loss=adv_loss, adv_acc1=adv_acc1)
                else:
                    logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                                  hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1)
            # if api is not None:
            #     hapi_acc1, hapi_acc5 = accuracy_fn(
            #             _output, hapi_label, num_classes=new_num_classes, topk=(1, 5))
            #     gt_acc1, gt_acc5 = accuracy_fn(
            #         _output, _label, num_classes=new_num_classes, topk=(1, 5))
            #     logger.update(n=batch_size,  gt_loss=float(gt_loss), gt_acc1=gt_acc1,
            #               hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1)
            # else:
            #     hapi_acc1, hapi_acc5 = accuracy_fn(
            #            _output, hapi_label, num_classes=new_num_classes, topk=(1, 5))
            #     tt,tf,ft,ff = missclassification_fn(_output, _label, hapi_label,new_num_classes)
            #     gt_acc1, gt_acc5 = accuracy_fn(
            #         _output, _label, num_classes=new_num_classes, topk=(1, 5))
            #     logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
            #               hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1)
                # logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                #           hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1,tt=tt,tf=tf,ft=ft,ff=ff)
    # if api is not None:
    if encoder_train:
        loss = (logger.meters['loss'].global_avg)
        if writer is not None and _epoch is not None and main_tag:
            from torch.utils.tensorboard import SummaryWriter
            assert isinstance(writer, SummaryWriter)
            writer.add_scalars(main_tag='loss/' + main_tag,
                               tag_scalar_dict={tag: loss}, global_step=_epoch)
        return loss

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

    # else:
    #     gt_loss, gt_acc1, hapi_loss, hapi_acc1,tt,tf,ft,ff = (logger.meters['gt_loss'].global_avg,
    #                 logger.meters['gt_acc1'].global_avg,
    #                 logger.meters['hapi_loss'].global_avg,
    #                 logger.meters['hapi_acc1'].global_avg,
    #                 logger.meters['tt'].global_avg,
    #                 logger.meters['tf'].global_avg,
    #                 logger.meters['ft'].global_avg,
    #                 logger.meters['ff'].global_avg)

    #     if writer is not None and _epoch is not None and main_tag:
    #         from torch.utils.tensorboard import SummaryWriter
    #         assert isinstance(writer, SummaryWriter)
    #         writer.add_scalars(main_tag='gt_loss/' + main_tag,
    #                     tag_scalar_dict={tag: gt_loss}, global_step=_epoch)
    #         writer.add_scalars(main_tag='gt_acc1/' + main_tag,
    #                     tag_scalar_dict={tag: gt_acc1}, global_step=_epoch)
    #         writer.add_scalars(main_tag='hapi_loss/' + main_tag,
    #                     tag_scalar_dict={tag: hapi_loss}, global_step=_epoch)
    #         writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
    #                     tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch)
    #         writer.add_scalars(main_tag='tt/' + main_tag,
    #                         tag_scalar_dict={tag: tt}, global_step=_epoch)
    #         writer.add_scalars(main_tag='tf/' + main_tag,
    #                         tag_scalar_dict={tag: tf}, global_step=_epoch)
    #         writer.add_scalars(main_tag='ft/' + main_tag,
    #                         tag_scalar_dict={tag: ft}, global_step=_epoch)
    #         writer.add_scalars(main_tag='ff/' + main_tag,
    #                         tag_scalar_dict={tag: ff}, global_step=_epoch)

    return hapi_acc1, hapi_loss


def attack_validate(module: nn.Module, num_classes: int,
                    loader: torch.utils.data.DataLoader,
                    print_prefix: str = 'Validate', indent: int = 0,
                    verbose: bool = True,
                    writer=None, main_tag: str = 'valid',
                    tag: str = '', _epoch: int = None,
                    label_train=False, hapi_label_train=False, api=False, task=None, after_loss_fn=None, adv_valid=False,
                    **kwargs) -> tuple[float, float]:
    r"""Evaluate the model.

    Returns:
        (float, float): Accuracy and loss.
    """
    module.eval()

    forward_fn = module.__call__

    logger = MetricLogger()
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

                    _input, _label, _soft_label, hapi_label = data
                    _input = _input.cuda()
                    _soft_label = _soft_label.cuda()
                    _label = _label.cuda()
                    hapi_label = hapi_label.cuda()
                    _output = forward_fn(_input)
                    if adv_valid:
                        loss, adv_x, adv_api_soft_label, adv_api_hapi_label = after_loss_fn(
                            forward_fn, _input=_input, _label=_label, _output=_output, mode='valid')
                        adv_x = adv_x.cuda()
                        adv_api_soft_label = adv_api_soft_label.cuda()
                        adv_output = forward_fn(adv_x)
                        adv_loss = float(loss_fn(_soft_label=adv_api_soft_label, _output=adv_output, **kwargs))

                case 'sentiment':
                    # input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label  = data
                    input_ids, attention_mask, _label, _soft_label, hapi_label = data
                    input_ids = input_ids.cuda()
                    # token_type_ids = token_type_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    _label = _label.cuda()
                    _soft_label = _soft_label.cuda()
                    hapi_label = hapi_label.cuda()

                    _output = forward_fn(input_ids=input_ids, attention_mask=attention_mask)
                    if adv_valid:
                        raise NotImplementedError(f'{adv_valid=} is not supported on sentiment yet.')

            gt_loss = float(loss_fn(_label=_label, _output=_output, **kwargs))
            if label_train:
                hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))
            elif hapi_label_train:
                hapi_loss = float(loss_fn(_soft_label=hapi_label, _output=_output, **kwargs))
            else:
                hapi_loss = float(loss_fn(_soft_label=_soft_label, _output=_output, **kwargs))

            batch_size = int(_label.size(0))
            match task:
                case 'sentiment':
                    _output = _output[:, :2]
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

                logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                              hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1,
                              adv_loss=adv_loss, adv_acc1=adv_acc1)
            else:
                logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                              hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1)
        else:
            with torch.no_grad():
                match task:
                    case 'emotion':

                        _input, _label, _soft_label, hapi_label = data
                        _input = _input.cuda()
                        _soft_label = _soft_label.cuda()
                        _label = _label.cuda()
                        hapi_label = hapi_label.cuda()
                        _output = forward_fn(_input)

                    case 'sentiment':
                        # input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label  = data
                        input_ids, attention_mask, _label, _soft_label, hapi_label = data
                        input_ids = input_ids.cuda()
                        # token_type_ids = token_type_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        _label = _label.cuda()
                        _soft_label = _soft_label.cuda()
                        hapi_label = hapi_label.cuda()

                        _output = forward_fn(input_ids=input_ids, attention_mask=attention_mask)
                        if adv_valid:
                            raise NotImplementedError(f'{adv_valid=} is not supported on sentiment yet.')

                gt_loss = float(loss_fn(_label=_label, _output=_output, **kwargs))
                if label_train:
                    hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))
                elif hapi_label_train:
                    hapi_loss = float(loss_fn(_soft_label=hapi_label, _output=_output, **kwargs))
                else:
                    hapi_loss = float(loss_fn(_soft_label=_soft_label, _output=_output, **kwargs))

                batch_size = int(_label.size(0))
                match task:
                    case 'sentiment':
                        _output = _output[:, :2]
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

                    logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
                                  hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1,
                                  adv_loss=adv_loss, adv_acc1=adv_acc1)
                else:
                    logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1,
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
        gt_loss, gt_acc1, hapi_loss, hapi_acc1, tt, tf, ft, ff = (logger.meters['gt_loss'].global_avg,
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


def train(module: nn.Module, num_classes: int,
          epochs: int, optimizer, lr_scheduler,
          log_dir: str = 'runs/test',
          grad_clip: float = 5.0,
          print_prefix: str = 'Distill', start_epoch: int = 0, resume: int = 0,
          validate_interval: int = 1, save: bool = True,
          loader_train: torch.utils.data.DataLoader = None,
          loader_valid: torch.utils.data.DataLoader = None,
          file_path: str = None,
          folder_path: str = None, suffix: str = None,
          main_tag: str = 'train', tag: str = '',

          verbose: bool = True, output_freq: str = 'iter', indent: int = 0,
          change_train_eval: bool = True, lr_scheduler_freq: str = 'epoch',
          backward_and_step: bool = True,
          mixmatch: bool = False,
          **kwargs):
    r"""Train the model"""
    if epochs <= 0:
        return

    forward_fn = module.__call__

    writer = SummaryWriter(log_dir=log_dir)
    validate_fn = train_validate

    scaler: torch.cuda.amp.GradScaler = None

    best_validate_result = (0.0, float('inf'))
    best_acc = 0.0
    if validate_interval != 0:
        best_validate_result = validate_fn(module=module, loader=loader_valid,
                                           writer=None, tag=tag, _epoch=start_epoch,
                                           verbose=verbose, indent=indent, num_classes=num_classes, **kwargs)
        best_acc = best_validate_result[0]

    params: list[nn.Parameter] = []
    for param_group in optimizer.param_groups:
        params.extend(param_group['params'])
    len_loader_train = len(loader_train)
    total_iter = (epochs - resume) * len_loader_train

    logger = MetricLogger()
    if mixmatch:
        logger.create_meters(loss=None)
    else:
        logger.create_meters(gt_loss=None, gt_acc1=None,
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

        # if _epoch < 10000:
        #     mode = 'train_STU' #kl loss / return raw data
        #     print(_epoch,mode)
        # elif _epoch >= 10000:
        #     mode = 'train_ADV_STU'  #kl loss / return adv data
        #     print(_epoch,mode)

        for i, data in enumerate(loader_epoch):
            _iter = _epoch * len_loader_train + i

            if mixmatch:
                print('no mixmatch support')
                # input_ids, token_type_ids, attention_mask, label, soft_label, hapi_label  = data

                # mixed_input = list(torch.split(mixed_input, batch_size))
                # mixed_input = interleave_fn(mixed_input, batch_size)

                # logits = [forward_fn(mixed_input[0])]
                # for input in mixed_input[1:]:
                #     logits.append(forward_fn(input))

                # # put interleaved samples back
                # logits = interleave_fn(logits, batch_size)
                # logits_x = logits[0]
                # logits_u = torch.cat(logits[1:], dim=0)

                # loss = loss_fn(outputs_x = logits_x, targets_x = mixed_target[:batch_size], outputs_u = logits_u, targets_u = mixed_target[batch_size:], iter = _iter)

            else:
                input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label = data
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                _label = _label.cuda()
                _soft_label = _soft_label.cuda()
                hapi_label = hapi_label.cuda()

                _output = forward_fn(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = loss_fn(_label=_label, _output=_output)

            if backward_and_step:
                optimizer.zero_grad()
                # backward the weights
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

            if lr_scheduler and lr_scheduler_freq == 'iter':
                lr_scheduler.step()

            if mixmatch:
                logger.update(n=batch_size, loss=float(loss))
            else:
                hapi_acc1, hapi_acc5 = accuracy_fn(
                    _output, hapi_label, num_classes=num_classes, topk=(1, 5))
                gt_acc1, gt_acc5 = accuracy_fn(
                    _output, _label, num_classes=num_classes, topk=(1, 5))
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
            loss = (logger.meters['loss'].global_avg)
            if writer is not None:
                writer.add_scalars(main_tag='loss/' + main_tag,
                                   tag_scalar_dict={tag: loss}, global_step=_epoch + start_epoch)
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
                                          **kwargs)
            cur_acc = validate_result[0]
            if cur_acc >= 0:
                best_validate_result = validate_result
                if verbose:
                    prints('{purple}best result update!{reset}'.format(
                        **ansi), indent=indent)
                    prints(f'Current Acc: {cur_acc:.3f}    '
                           f'Previous Best Acc: {best_acc:.3f}',
                           indent=indent)
                best_acc = cur_acc
                if save:
                    save_fn(file_path=file_path, folder_path=folder_path,
                            suffix=suffix, verbose=verbose)
            if verbose:
                prints('-' * 50, indent=indent)
    module.zero_grad()
    return best_validate_result


def train_validate(module: nn.Module, num_classes: int,
                   loader: torch.utils.data.DataLoader,
                   print_prefix: str = 'Validate', indent: int = 0,
                   verbose: bool = True,
                   writer=None, main_tag: str = 'valid',
                   tag: str = '', _epoch: int = None,
                   **kwargs) -> tuple[float, float]:
    r"""Evaluate the model.

    Returns:
        (float, float): Accuracy and loss.
    """
    module.eval()

    forward_fn = module.__call__

    logger = MetricLogger()
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

        input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label = data
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        _label = _label.cuda()
        _soft_label = _soft_label.cuda()
        hapi_label = hapi_label.cuda()

        with torch.no_grad():
            _output = forward_fn(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            gt_loss = float(loss_fn(_label=_label, _output=_output, **kwargs))

            hapi_loss = float(loss_fn(_label=hapi_label, _output=_output, **kwargs))

            hapi_acc1, hapi_acc5 = accuracy_fn(
                _output, hapi_label, num_classes=num_classes, topk=(1, 5))
            gt_acc1, gt_acc5 = accuracy_fn(
                _output, _label, num_classes=num_classes, topk=(1, 5))
            batch_size = int(_label.size(0))
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

    return hapi_acc1, hapi_loss
