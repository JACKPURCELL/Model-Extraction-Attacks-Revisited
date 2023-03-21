import torch.nn as nn
import torch
class MyModule(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(MyModule, self).__init__()
        self.b =torch.tensor(100)

    def forward(self, x):

        return x+self.b


MyModule_ = MyModule()

forward_fn =MyModule_.__call__

class PGD:
    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

PGD_ = PGD(forward_fn)       

print(PGD_.forward_fn(torch.tensor(1)))
MyModule_.b =200
print(PGD_.forward_fn(torch.tensor(1)))