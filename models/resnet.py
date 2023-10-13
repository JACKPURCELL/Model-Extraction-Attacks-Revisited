#!/usr/bin/env python3


from collections import OrderedDict
import pickle
import torch
import torch.nn as nn
import torchvision.models
from models.recorder import EmbeddingRecorder
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data
from typing import Generator, Iterator, Mapping
from collections.abc import Iterable
from adversirial.pgd import PGD
from optimizer.lion import Lion
from torchvision import transforms
from .resnet_cifar10 import *
        
class ResNet(nn.Module):

    def __init__(self, norm_par=None,model_name: str = 'resnet50',record_embedding=False,num_classes=7,path=None):

        super(ResNet, self).__init__() 
        if model_name == 'resnet50_vggface2':
            ModelClass = getattr(torchvision.models, 'resnet50')
            _model = ModelClass(weights='DEFAULT').cuda()
            _model.fc = nn.Linear(in_features=_model.fc.in_features, out_features=8631,bias=True).cuda()
            _model = self.load_state_dict_vgg(_model,'/home/jkl6486/hermes/resnet50_ft_weight.pkl')
        elif 'raw' in model_name:
            model_name = model_name.split('_raw')[0]
            ModelClass = getattr(torchvision.models, model_name)
            _model = ModelClass().cuda()
        elif 'cifar10' in model_name:
            ModelClass = getattr(torchvision.models, model_name.split('_')[0])
            _model = ModelClass().cuda()
            _model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False).cuda()
        else:
            ModelClass = getattr(torchvision.models, model_name)
            _model = ModelClass(weights='DEFAULT').cuda()

            
       
        if norm_par is not None:
            self.norm_par = norm_par
            self.transform = transforms.Normalize( mean=norm_par['mean'], std= norm_par['std'])   
        _model.fc = nn.Linear(in_features=_model.fc.in_features, out_features=num_classes,bias=True).cuda()
        module_list: list[nn.Module] = []
        module_list.append(('conv1', _model.conv1))
        module_list.append(('bn1', _model.bn1))
        module_list.append(('relu', _model.relu))
        module_list.append(('maxpool', _model.maxpool))
        module_list.extend([('layer1', _model.layer1),
                            ('layer2', _model.layer2),
                            ('layer3', _model.layer3),
                            ('layer4', _model.layer4)])
        
        self.features = nn.Sequential(OrderedDict(module_list)).cuda()
        module_list: list[nn.Module] = []
        module_list.extend([
                            ('fc', nn.Linear(in_features=_model.fc.in_features, out_features=num_classes,bias=True))
                         ])
        self.classifier = nn.Sequential(OrderedDict(module_list)).cuda()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).cuda()
        self.embedding_recorder = EmbeddingRecorder(record_embedding).cuda()
        
        if path is not None:
            self.load_state_dict(path)

    def load_state_dict_vgg(self, _model,fname):
        """
        Set parameters converted from Caffe models authors of VGGFace2 provide.
        See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
        Arguments:
            model: model
            fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
        """
        with open(fname, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')

        own_state = _model.state_dict()
        for name, param in weights.items():
            if name in own_state:
                try:
                    own_state[name].copy_(torch.from_numpy(param))
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                    'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            else:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))

        return _model
    
    def load_state_dict(self, fname):
        """
        Set parameters converted from Caffe models authors of VGGFace2 provide.
        See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
        Arguments:
            model: model
            fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
        """

        weights = torch.load(fname)

        own_state = self.features.state_dict()
        for name, param in weights.items():
            if 'features' in name:
                name = name.split('features.')[1]
                if name in own_state:
                    try:
                        own_state[name].copy_(param)
                    except Exception:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                        'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
                else:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        own_state = self.classifier.state_dict()
        for name, param in weights.items():
            if 'classifier' in name:
                name = name.split('classifier.')[1]
                if name in own_state:
                    try:
                        own_state[name].copy_(param)
                    except Exception:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                        'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
                else:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))
            

        
    def forward(self, x):
        if self.norm_par is not None:
            x = self.transform(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding_recorder(x)
        x = self.classifier(x)
        return x

    def get_usl(self,x):
        if self.norm_par is not None:
            x = self.transform(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
    def get_features(self, x):
        if self.norm_par is None:
            return self.features(x)
        return self.features(self.transform(x))
    
    def define_optimizer(
        self, parameters: str | Iterator[nn.Parameter] = 'full',
        OptimType: str | type[Optimizer] = 'Lion',
        lr: float = 3e-4, custom_lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 1.0,
        lr_scheduler: bool = False,
        lr_scheduler_type: str = 'CosineAnnealingLR',
        lr_step_size: int = 30, lr_gamma: float = 0.1,
        epochs: int = None, lr_min: float = 0.0,
        lr_warmup_percent: float = 0.0, lr_warmup_method: str = 'constant',
        lr_warmup_epoch=0,
        lr_warmup_decay: float = 0.01,
        betas = (0.9, 0.99),
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

            case 'full':
                kwargs['momentum'] = momentum
                kwargs['weight_decay'] = weight_decay
                kwargs['betas'] = betas
                kwargs['eps'] = eps
                keys = OptimType.__init__.__code__.co_varnames
                kwargs = {k: v for k, v in kwargs.items() if k in keys}
                params =  list(self.parameters())
                # params.extend(list(self.classifier.parameters()))
                optimizer = OptimType(params, lr, **kwargs)
            case _:
                raise ValueError        

        _lr_scheduler: _LRScheduler = None
        
        lr_warmup_epochs = lr_warmup_epoch if lr_warmup_epoch != 0 else int(epochs * lr_warmup_percent)
        
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
                            optimizer, factor=1.0,
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
