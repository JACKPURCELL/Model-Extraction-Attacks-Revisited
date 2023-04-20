#!/usr/bin/env python3

from optimizer.pgd import PGDoptimizer

from utils.output import prints, ansi
from utils.logger import SmoothedValue
from utils.tensor import repeat_to_batch

from dataset.dataset import split_dataset


import torch
import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.utils.data import DataLoader
from utils.model import loss_fn as loss_fn_model
def get_func_key(func: Callable[..., torch.Tensor]) -> str:
    keys = func.__code__.co_varnames
    valid_keys = [key for key in keys if 'target' in key or 'label' in key or 'y' in key or 'Y' in key]
    assert len(valid_keys) == 1, keys
    return valid_keys[0]





class PGD(PGDoptimizer):
    r"""PGD Adversarial Attack.

    Args:
        pgd_alpha (float): learning rate :math:`\alpha`. Default: :math:`\frac{3}{255}`.
        pgd_eps (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{8}{255}`.
    """

    name: str = 'pgd'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--pgd_alpha', type=float, help='PGD learning rate per step, defaults to 2.0/255')
        group.add_argument('--pgd_eps', type=float, help='Projection norm constraint, defaults to 8.0/255')
        group.add_argument('--iteration', type=int, help='Attack Iteration, defaults to 7')
        group.add_argument('--stop_threshold', type=float, help='early stop confidence, defaults to 0.99')
        group.add_argument('--target_class', type=int, help='Do not set it if using target_idx')
        group.add_argument('--target_idx', type=int,
                           help='Target label order in original classification, defaults to -1 '
                           '(0 for untargeted attack, 1 for most possible class, -1 for most unpossible class)')
        group.add_argument('--test_num', type=int, help='total number of test examples for PGD, defaults to 1000.')
        group.add_argument('--num_restart', type=int,
                           help='number of random init for PGD, defaults to 0 (without random initialization).')
        group.add_argument('--require_class', action='store_true',
                           help='Require class to be desired in early_stop_check.')

        group.add_argument('--grad_method', choices=['white', 'nes', 'sgd', 'hess', 'zoo'],
                           help='gradient estimation method, defaults to "white"')
        group.add_argument('--query_num', type=int,
                           help='query numbers for black box gradient estimation, defaults to 100.')
        group.add_argument('--sigma', type=float,
                           help='gaussian sampling std for black box gradient estimation, defaults to 1e-3')
        return group

    def __init__(self,forward_fn, validset,target_class: int = None, target_idx: int = -1, test_num: int = 1000, num_restart: int = 0,
                 require_class: bool = False, **kwargs):
        self.target_class = target_class
        self.target_idx = target_idx
        self.test_num = test_num
        self.num_restart = num_restart
        self.require_class = require_class
        kwargs.update(random_init=bool(num_restart))
        super().__init__(**kwargs)
        self.param_list['pgd_attack'] = ['target_class', 'target_idx', 'test_num', 'num_restart', 'require_class']
        self.forward_fn=forward_fn
        self.validset=validset
        self.softmax = nn.Softmax(dim=1)
    
    @torch.no_grad()
    def remove_misclassify(self, data: tuple[torch.Tensor, torch.Tensor],
                           **kwargs):
        r"""Remove misclassified samples in a data batch.

        Args:
            data (tuple[torch.Tensor, torch.Tensor]):
                The input and label to process with shape ``(N, *)`` and ``(N)``.
            **kwargs: Keyword arguments passed to :meth:`get_data`.

        Returns:
            (torch.Tensor, torch.Tensor):
                The processed input and label with shape ``(N - k, *)`` and ``(N - k)``.
        """
        _input, _label = self.get_data(data, **kwargs)
        _classification = self.get_class(_input)
        repeat_idx = _classification.eq(_label)
        return _input[repeat_idx], _label[repeat_idx]
    
    @torch.no_grad()
    def generate_target(self,  _input: torch.Tensor,
                        idx: int = 1, same: bool = False
                        ) -> torch.Tensor:
        r"""Generate target labels of a batched input based on
            the classification confidence ranking index.

        Args:
            module (torch.nn.Module): The module to process.
            _input (torch.Tensor): The input tensor.
            idx (int): The classification confidence
                rank of target class.
                Defaults to ``1``.
            same (bool): Generate the same label
                for all samples using mod.
                Defaults to ``False``.

        Returns:
            torch.Tensor:
                The generated target label with shape ``(N)``.
        """
        _output: torch.Tensor = self.forward_fn(_input)
        target = _output.argsort(dim=-1, descending=True)[:, idx]
        if same:
            target = repeat_to_batch(target.mode(dim=0)[0], len(_input))
        return target


    def get_prob(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Get the probability classification vector of :attr:`_input`.

        Args:
            _input (torch.Tensor): The batched input tensor
                passed to :meth:`_Model.get_logits()`.
            **kwargs: Keyword arguments passed to :meth:`get_logits()`.

        Returns:
            torch.Tensor: The probability tensor with shape ``(N, C)``.
        """
        return self.softmax(self.forward_fn(_input, **kwargs))

    def get_target_prob(self, _input: torch.Tensor,
                        target: int | list[int] | torch.Tensor,
                        **kwargs) -> torch.Tensor:
        r"""Get the probability w.r.t. :attr:`target` class of :attr:`_input`
        (using :any:`torch.gather`).

        Args:
            _input (torch.Tensor): The batched input tensor
                passed to :meth:`_Model.get_logits()`.
            target (int | list[int] | torch.Tensor): Batched target classes.
            **kwargs: Keyword arguments passed to :meth:`get_logits()`.

        Returns:
            torch.Tensor: The probability tensor with shape ``(N)``.
        """
        match target:
            case int():
                target = [target] * len(_input)
            case list():
                target = torch.tensor(target, device=_input.device)
        return self.get_prob(_input, **kwargs).gather(
            dim=1, index=target.unsqueeze(1)).flatten()

    def get_class(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Get the class classification result of :attr:`_input`
        (using :any:`torch.argmax`).

        Args:
            _input (torch.Tensor): The batched input tensor
                passed to :meth:`_Model.get_logits()`.
            **kwargs: Keyword arguments passed to :meth:`get_logits()`.

        Returns:
            torch.Tensor: The classes tensor with shape ``(N)``.
        """
        return self.forward_fn(_input, **kwargs).argmax(dim=-1)


    def attack(self, verbose: int = 1, **kwargs) -> tuple[float, float]:
        # 
        validset = self.validset
        testset, _ = split_dataset(validset, percent=0.3)
        loader = DataLoader(dataset=testset,batch_size=self.batch_size,
                    shuffle=True,drop_last=True)
        fmt_str = '{global_avg:7.3f} ({min:7.3f}  {max:7.3f})'
        total_adv_target_conf = SmoothedValue(fmt=fmt_str)
        total_org_target_conf = SmoothedValue(fmt=fmt_str)
        succ_adv_target_conf = SmoothedValue(fmt=fmt_str)

        total_adv_org_conf = SmoothedValue(fmt=fmt_str)
        total_org_org_conf = SmoothedValue(fmt=fmt_str)
        succ_adv_org_conf = SmoothedValue(fmt=fmt_str)

        total_iter_list = SmoothedValue(fmt=fmt_str)
        succ_iter_list = SmoothedValue(fmt=fmt_str)

        succ_idx_list: list[int] = []
        for data in loader:
            rest_length = self.test_num - total_adv_target_conf.count
            if rest_length <= 0:
                break
            #移除错误分类的部分
            _input, _label = self.remove_misclassify(data)
            if len(_label) == 0:
                continue

            if len(_label) > rest_length:
                _input = _input[:rest_length]
                _label = _label[:rest_length]

                #    _output: torch.Tensor = module(_input)
    # target = _output.argsort(dim=-1, descending=True)[:, idx]
            target = self.generate_target(_input, idx=self.target_idx) if self.target_class is None \
                else self.target_class * torch.ones_like(_label)
            adv_input = _input.clone().detach()
            iter_list = -torch.ones(len(_label), dtype=torch.long)
            current_idx = torch.arange(len(iter_list))
            for _ in range(max(self.num_restart, 1)):
                #生成对抗样本
                temp_adv_input, temp_iter_list = self.optimize(_input[current_idx],
                                                               target=target[current_idx], **kwargs)
                #存储对抗样本到list                                               
                adv_input[current_idx] = temp_adv_input
                iter_list[current_idx] = temp_iter_list
                fail_idx = iter_list == -1
                if (~fail_idx).all():
                    break
                current_idx = current_idx[fail_idx]
            for i, _iter in enumerate(iter_list):
                if _iter != -1:
                    succ_idx_list.append(total_iter_list.count + i)
            adv_target_conf = self.get_target_prob(adv_input, target)
            adv_org_conf = self.get_target_prob(adv_input, _label)
            org_target_conf = self.get_target_prob(_input, target)
            org_org_conf = self.get_target_prob(_input, _label)

            total_adv_target_conf.update_list(adv_target_conf.detach().cpu().tolist())
            total_adv_org_conf.update_list(adv_org_conf.detach().cpu().tolist())
            succ_adv_target_conf.update_list(adv_target_conf[iter_list != -1].detach().cpu().tolist())
            succ_adv_org_conf.update_list(adv_org_conf[iter_list != -1].detach().cpu().tolist())
            total_org_target_conf.update_list(org_target_conf.detach().cpu().tolist())
            total_org_org_conf.update_list(org_org_conf.detach().cpu().tolist())

            total_iter_list.update_list(torch.where(iter_list != -1, iter_list, 2 *
                                        self.iteration * torch.ones_like(iter_list)).tolist())
            succ_iter_list.update_list(iter_list[iter_list != -1].tolist())
            if verbose >= 3:
                prints(f'{ansi["green"]}{succ_iter_list.count} / {total_iter_list.count}{ansi["reset"]}')
            if verbose >= 4:
                prints(f'{total_iter_list=:}', indent=4)
                prints(f'{succ_iter_list=:}', indent=4)
                prints()
                prints('-------------------------------------------------', indent=4)
                prints(f'{ansi["yellow"]}Target Class:{ansi["reset"]}', indent=4)
                prints(f'{total_adv_target_conf=:}', indent=8)
                prints(f'{total_org_target_conf=:}', indent=8)
                prints(f'{succ_adv_target_conf=:}', indent=8)
                prints()
                prints('-------------------------------------------------', indent=4)
                prints(f'{ansi["yellow"]}Original Class:{ansi["reset"]}', indent=4)
                prints(f'{total_adv_org_conf=:}', indent=8)
                prints(f'{total_org_org_conf=:}', indent=8)
                prints(f'{succ_adv_org_conf=:}', indent=8)
        if verbose:
            prints(f'{ansi["green"]}{succ_iter_list.count} / {total_iter_list.count}{ansi["reset"]}')
            prints(succ_idx_list)
        if verbose >= 2:
            prints(f'{total_iter_list=:}', indent=4)
            prints(f'{succ_iter_list=:}', indent=4)
            prints()
            prints('-------------------------------------------------', indent=4)
            prints(f'{ansi["yellow"]}Target Class:{ansi["reset"]}', indent=4)
            prints(f'{total_adv_target_conf=:}', indent=8)
            prints(f'{total_org_target_conf=:}', indent=8)
            prints(f'{succ_adv_target_conf=:}', indent=8)
            prints()
            prints('-------------------------------------------------', indent=4)
            prints(f'{ansi["yellow"]}Original Class:{ansi["reset"]}', indent=4)
            prints(f'{total_adv_org_conf=:}', indent=8)
            prints(f'{total_org_org_conf=:}', indent=8)
            prints(f'{succ_adv_org_conf=:}', indent=8)
        return float(succ_iter_list.count) / total_iter_list.count, total_iter_list.global_avg

    def optimize(self, _input: torch.Tensor, *args,
                 target: None | int | torch.Tensor = None, target_idx: int = None,
                 loss_fn: Callable[..., torch.Tensor] = None,
                 require_class: bool = None,
                 loss_kwargs: dict[str, torch.Tensor] = {},
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if len(_input) == 0:
            return _input, None
        target_idx = self.target_idx if target_idx is None else target_idx
        match target:
            case None:
                target = self.generate_target(_input, idx=self.target_idx) if self.target_class is None \
                    else self.target_class * torch.ones(len(_input), dtype=torch.long, device=_input.device)
            case int():
                target = target * torch.ones(len(_input), dtype=torch.long, device=_input.device)
        if loss_fn is None and self.loss_fn is None:
            untarget_condition = self.target_class is None and self.target_idx == 0

            def _loss_fn(target: torch.Tensor, reduction: str = 'mean', **kwargs):

                if len(target.shape) == 1:
                    loss = loss_fn_model( _output=kwargs['_output'],_label=target, reduction=reduction)
                else:
                    loss = loss_fn_model( _output=kwargs['_output'],_soft_label=target, reduction=reduction)
                return -loss if untarget_condition else loss
            loss_fn = _loss_fn
        loss_kwargs.update(target=target)
        return super().optimize(_input, *args, target=target,
                                loss_fn=loss_fn, require_class=require_class,
                                loss_kwargs=loss_kwargs,
                                **kwargs)

    @torch.no_grad()
    def early_stop_check(self, current_idx: torch.Tensor,
                         adv_input: torch.Tensor, target: torch.Tensor, *args,
                         stop_threshold: float = None, require_class: bool = None,
                         **kwargs) -> torch.Tensor:
        stop_threshold = stop_threshold if stop_threshold is not None else self.stop_threshold
        require_class = True
        # require_class = require_class if require_class is not None else self.require_class
        if self.stop_threshold is None:
            return torch.zeros(len(current_idx), dtype=torch.bool)
        _confidence = self.get_target_prob(adv_input[current_idx], target[current_idx])
        untarget_condition = self.target_class is None and self.target_idx == 0
        result = _confidence > stop_threshold
        if untarget_condition:
            result = ~result
        if require_class:
            _class = self.get_class(adv_input[current_idx])
            class_result = _class == target[current_idx]
            if untarget_condition:
                class_result = ~class_result
            #True mean attack successful
            result = result.bitwise_and(class_result)
        return result

    @torch.no_grad()
    def output_info(self, org_input: torch.Tensor, noise: torch.Tensor, target: torch.Tensor,
                    loss_fn: Callable[[torch.Tensor], torch.Tensor] = None, **kwargs):
        super().output_info(org_input=org_input, noise=noise, loss_fn=loss_fn, **kwargs)
        # prints('Original class     : ', _label, indent=self.indent)
        # prints('Original confidence: ', _confidence, indent=self.indent)
        _confidence = self.get_target_prob(org_input + noise, target)
        prints('Target   class     : ', target.detach().cpu().tolist(), indent=self.indent)
        prints('Target   confidence: ', _confidence.detach().cpu().tolist(), indent=self.indent)
