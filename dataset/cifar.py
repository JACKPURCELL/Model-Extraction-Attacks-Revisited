#!/usr/bin/env python3


import torch
from torchvision import transforms


import torchvision.datasets as datasets
import torch.nn as nn

def cutout(img: torch.Tensor, length: int | tuple[int, int] | torch.Tensor,
           fill_values: float | torch.Tensor = 0.0) -> torch.Tensor:
    if isinstance(length, int):
        length = (length, length)
    h, w = img.size(-2), img.size(-1)
    mask = torch.ones(h, w, dtype=torch.bool, device=img.device)
    device = length.device if isinstance(length, torch.Tensor) else img.device
    y = torch.randint(0, h, [1], device=device)
    x = torch.randint(0, w, [1], device=device)
    first_half = [length[0] // 2, length[1] // 2]
    second_half = [length[0] - first_half[0], length[1] - first_half[1]]

    y1 = max(y - first_half[0], 0)
    y2 = min(y + second_half[0], h)
    x1 = max(x - first_half[1], 0)
    x2 = min(x + second_half[1], w)
    mask[y1: y2, x1: x2] = False
    return mask * img + ~mask * fill_values


class Cutout(nn.Module):
    def __init__(self, length: int,
                 fill_values: float | torch.Tensor = 0.0):
        super().__init__()
        self.length = length
        self.fill_values = fill_values

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return cutout(img, self.length, self.fill_values)
    
def get_transform_cifar(mode: str, auto_augment: bool = False,
                        cutout: bool = False, cutout_length: int = None,
                        data_shape: list[int] = [3, 32, 32],norm_par=None,
                        ) -> transforms.Compose:
    if mode != 'train':
        return transforms.Compose([transforms.PILToTensor(),
                                   transforms.ConvertImageDtype(torch.float)])
    cutout_length = cutout_length or data_shape[-1] // 2
    transform = [
        transforms.RandomCrop(data_shape[-2:], padding=data_shape[-1] // 8),
        transforms.RandomHorizontalFlip(),
    ]
    if auto_augment:
        transform.append(transforms.AutoAugment(
            transforms.AutoAugmentPolicy.CIFAR10))
    transform.append(transforms.PILToTensor())
    transform.append(transforms.ConvertImageDtype(torch.float))
    if cutout:
        transform.append(Cutout(cutout_length))
        
    if norm_par is not None:
                transform.append(transforms.Normalize(
                mean=norm_par['mean'], std=norm_par['std']))
    return transforms.Compose(transform)

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
    
class CIFAR10(datasets.CIFAR10):
    r"""CIFAR10 dataset introduced by Alex Krizhevsky in 2009.
    It inherits :class:`trojanvision.datasets.ImageSet`.

    See Also:
        * torchvision: :any:`torchvision.datasets.CIFAR10`
        * paper: `Learning Multiple Layers of Features from Tiny Images`_
        * website: https://www.cs.toronto.edu/~kriz/cifar.html

    Attributes:
        name (str): ``'cifar10'``
        num_classes (int): ``10``
        data_shape (list[int]): ``[3, 32, 32]``
        class_names (list[str]):
            | ``['airplane', 'automobile', 'bird', 'cat', 'deer',``
            | ``'dog', 'frog', 'horse', 'ship', 'truck']``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.49139968, 0.48215827, 0.44653124],``
            | ``'std'  : [0.24703233, 0.24348505, 0.26158768]}``

    .. _Learning Multiple Layers of Features from Tiny Images:
        https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """
    name = 'cifar10'
    num_classes = 10
    data_shape = [3, 32, 32]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
# torchvision.datasets.CIFAR10(root: str, train: bool = True, 
#                              transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
    def __init__(self, mode, transform='Normal',**kwargs):
        if transform == 'Normal':
            transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
           
        ])
        elif transform == 'mixmatch':
            transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
           
        ])
            transform = TransformTwice(transform)
            
            # get_transform_cifar(
            #         mode, auto_augment=True,
            #         cutout=False, cutout_length=None,
            #         data_shape=self.data_shape,norm_par=None)
        elif transform == 'raw':
            transform = transforms.Compose([transforms.PILToTensor(),
                                   transforms.ConvertImageDtype(torch.float)])
        
        self.norm_par = {'mean': [0.49139968, 0.48215827, 0.44653124],
                         'std': [0.24703233, 0.24348505, 0.26158768] }
        
        super().__init__(root='/data/jc/data/image/cifar10',train=(mode == 'train'),transform=transform,download=True)

    
