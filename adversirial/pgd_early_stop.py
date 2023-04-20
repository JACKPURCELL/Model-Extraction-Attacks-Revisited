from torchattacks import PGD
import torch
import torch.nn as nn
class PGD_early(PGD):
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True):
        super().__init__(model, eps, alpha, steps, random_start)
        self.stop_threshold = 1.0
        self.require_class = True
        self.target_class = None
        self.forward_fn = None
        self.softmax = nn.Softmax(dim=-1)
    
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
        untarget_condition = self.target_class is None 
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
    
    def forward(self, forward_fn,images, labels):
        self._check_inputs(images)
        self.forward_fn=forward_fn
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            
        iter_list: torch.Tensor = torch.zeros(len(adv_images), dtype=torch.long).cuda()
        current_idx = torch.arange(len(iter_list)).cuda()
        
        for _ in range(self.steps):
            
            early_stop_result = self.early_stop_check(
                current_idx=current_idx,target=labels,
                adv_input=adv_images, org_input=images)
            not_early_stop_result = ~early_stop_result
            current_idx = current_idx[not_early_stop_result]
            iter_list[current_idx] += 1
            if early_stop_result.all():
                return adv_images.detach(), iter_list
            
            x = adv_images[current_idx]
            x.requires_grad = True
            outputs = self.get_logits(x)

            if self.targeted:
                cost = -loss(outputs, target_labels[current_idx])
            else:
                cost = loss(outputs, labels[current_idx])

            grad = torch.autograd.grad(cost, x,
                                       retain_graph=False, create_graph=False)[0]

            x = x.detach() + self.alpha*grad.sign()
            delta = torch.clamp(x - images[current_idx],
                                min=-self.eps, max=self.eps)
            adv_images[current_idx] = torch.clamp(images[current_idx] + delta, min=0, max=1).detach()
            
        early_stop_result = self.early_stop_check(
        current_idx=current_idx,target=labels,
        adv_input=adv_images, org_input=images)
        current_idx = current_idx[~early_stop_result]
        iter_list[current_idx] += 1
        
        return (adv_images.detach(),
                torch.where(iter_list <= self.steps, iter_list,
                            -torch.ones_like(iter_list)))
        

        # return adv_images