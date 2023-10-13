#!/usr/bin/env python3


import torch
import torch.nn as nn
import torchvision.models

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data
from typing import Generator, Iterator, Mapping
from collections.abc import Iterable
from adversirial.pgd import PGD
from optimizer.lion import Lion
from torchvision import transforms

class efficientnet(nn.Module):

    def __init__(self, norm_par=None,model_name: str = 'efficientnet_v2_m',num_classes=7):

        super(efficientnet, self).__init__() 
        ModelClass = getattr(torchvision.models, model_name)
        if 'raw' in model_name:
            ModelClass = getattr(torchvision.models, model_name.split('_raw')[0])
            self.model = ModelClass().cuda()
            
        else:    
            ModelClass = getattr(torchvision.models, model_name)
            self.model = ModelClass(weights='DEFAULT').cuda()
        if norm_par is not None:
            self.norm_par = norm_par
            self.transform = transforms.Normalize( mean=norm_par['mean'], std= norm_par['std'])       
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes,bias=True).cuda()



    def forward(self, x):
        if self.norm_par is None:
            return self.model(x)
        return self.model(self.transform(x))
    
    def define_optimizer(
        self, parameters: str | Iterator[nn.Parameter] = 'full',
        OptimType: str | type[Optimizer] = 'Adam',
        lr: float = 0.1, custom_lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0,
        lr_scheduler: bool = False,
        lr_scheduler_type: str = 'CosineAnnealingLR',
        lr_step_size: int = 30, lr_gamma: float = 0.1,
        epochs: int = None, lr_min: float = 0.0,
        lr_warmup_percent: float = 0.0, lr_warmup_method: str = 'constant',
        lr_warmup_decay: float = 0.01,
        betas = (0.9, 0.999),
        eps = 1e-8,
        **kwargs) -> tuple[Optimizer, _LRScheduler]:
        r"""Define optimizer and lr_scheduler.

        Args:
            parameters (str | ~collections.abc.Iterable[torch.nn.parameter.Parameter]):
                The parameters to optimize while other model parameters are frozen.
                If :class:`str`, set :attr:`parameters` as:

                    * ``'features': self._model.features``
                    * ``'classifier' | 'partial': self._model.classifier``
                    * ``'full': self._model``

                Defaults to ``'full'``.
            OptimType (str | type[Optimizer]):
                The optimizer type.
                If :class:`str`, load from module :any:`torch.optim`.
                Defaults to ``'SGD'``.
            lr (float): The learning rate of optimizer. Defaults to ``0.1``.
            momentum (float): The momentum of optimizer. Defaults to ``0.0``.
            weight_decay (float): The momentum of optimizer. Defaults to ``0.0``.
            lr_scheduler (bool): Whether to enable lr_scheduler. Defaults to ``False``.
            lr_scheduler_type (str): The type of lr_scheduler.
                Defaults to ``'CosineAnnealingLR'``.

                Available lr_scheduler types (use string rather than type):

                    * :any:`torch.optim.lr_scheduler.StepLR`
                    * :any:`torch.optim.lr_scheduler.CosineAnnealingLR`
                    * :any:`torch.optim.lr_scheduler.ExponentialLR`
            lr_step_size (int): :attr:`step_size` for :any:`torch.optim.lr_scheduler.StepLR`.
                Defaults to ``30``.
            lr_gamma (float): :attr:`gamma` for :any:`torch.optim.lr_scheduler.StepLR`
                or :any:`torch.optim.lr_scheduler.ExponentialLR`.
                Defaults to ``0.1``.
            epochs (int): Total training epochs.
                ``epochs - lr_warmup_epochs`` is passed as :attr:`T_max`
                to any:`torch.optim.lr_scheduler.CosineAnnealingLR`.
                Defaults to ``None``.
            lr_min (float): The minimum of learning rate.
                It's passed as :attr:`eta_min`
                to any:`torch.optim.lr_scheduler.CosineAnnealingLR`.
                Defaults to ``0.0``.
            lr_warmup_epochs (int): Learning rate warmup epochs.
                Passed as :attr:`total_iters` to lr_scheduler.
                Defaults to ``0``.
            lr_warmup_method (str): Learning rate warmup methods.
                Choose from ``['constant', 'linear']``.
                Defaults to ``'constant'``.
            lr_warmup_decay (float): Learning rate warmup decay factor.
                Passed as :attr:`factor` (:attr:`start_factor`) to lr_scheduler.
                Defaults to ``0.01``.
            **kwargs: Keyword arguments passed to optimizer init method.

        Returns:
            (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler):
                The tuple of optimizer and lr_scheduler.
        """
        if isinstance(OptimType, str):
            if OptimType == 'Lion':
                OptimType = Lion
            else:
                OptimType: type[Optimizer] = getattr(torch.optim, OptimType)
        match parameters:
            case 'classifier' | 'partial':
                bert_identifiers = ['features']
                no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']
                grouped_model_parameters = [
                        {'params': [param for name, param in self.model.classifier.named_parameters()],
                        'lr': custom_lr,
                        'betas': betas,
                        'weight_decay': 0.0,
                        'eps': eps}
                ]
                optimizer = OptimType(grouped_model_parameters)
            case 'full':
                kwargs['momentum'] = momentum
                kwargs['weight_decay'] = weight_decay
                kwargs['betas'] = betas
                kwargs['eps'] = eps
                keys = OptimType.__init__.__code__.co_varnames
                kwargs = {k: v for k, v in kwargs.items() if k in keys}
                params = self.model.parameters()
                optimizer = OptimType(params, lr, **kwargs)
            case _:
                raise ValueError        

        _lr_scheduler: _LRScheduler = None
        
        lr_warmup_epochs = int(epochs * lr_warmup_percent)
        
        if lr_scheduler:
            main_lr_scheduler: _LRScheduler
            match lr_scheduler_type:
                case 'StepLR':
                    main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=lr_step_size, gamma=lr_gamma)
                case 'CosineAnnealingLR':
                    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=epochs - lr_warmup_epochs, eta_min=lr_min)
                case 'ExponentialLR':
                    main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer, gamma=lr_gamma)
                case _:
                    raise NotImplementedError(
                        f'Invalid {lr_scheduler_type=}.'
                        'Only "StepLR", "CosineAnnealingLR" and "ExponentialLR" '
                        'are supported.')
            if lr_warmup_epochs > 0:
                match lr_warmup_method:
                    case 'linear':
                        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                            optimizer, start_factor=lr_warmup_decay,
                            total_iters=lr_warmup_epochs)
                    case 'constant':
                        warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                            optimizer, factor=lr_warmup_decay,
                            total_iters=lr_warmup_epochs)
                    case _:
                        raise NotImplementedError(
                            f'Invalid {lr_warmup_method=}.'
                            'Only "linear" and "constant" are supported.')
                _lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                    milestones=[lr_warmup_epochs])
            else:
                _lr_scheduler = main_lr_scheduler
        return optimizer, _lr_scheduler
