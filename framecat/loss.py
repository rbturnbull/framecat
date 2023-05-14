import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma:float=2.0,
        weights:Tensor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.weights = weights

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """
        Adapted from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
        """
        target = target.view(-1,1)
        log_probabilities = F.log_softmax(predictions, dim=-1)
        log_probability = log_probabilities.gather(1,target)
        probability = Variable(log_probability.data.exp())
        loss = -(1-probability)** self.gamma * log_probability

        # Weights
        if self.weights is not None:
            self.weights = self.weights.to(target.device)
            loss *= torch.gather(self.weights, -1, target)

        return loss.mean()


class DateLossMSE(nn.Module):
    def forward(self, predictions: Tensor, target: Tensor):
        error_lower = torch.clamp_min(target[:,0]-predictions[:,0], 0.0)
        error_upper = torch.clamp_min(predictions[:,0] - target[:,1], 0.0)
        error = error_lower + error_upper
        return torch.mean(error**2)


class CatenaCombinedLoss(nn.Module):
    def __init__(self, gamma:float=0.0, date_weight:float=0.2, **kwargs):
        super().__init__(**kwargs)
        self.date_weight = date_weight
        self.date_loss = DateLossMSE()
        self.category_loss = FocalLoss(gamma=gamma) if gamma > 0.0 else nn.CrossEntropyLoss()

    def forward(self, predictions: Tensor, category_target: Tensor, date_range:Tensor):
        loss = (1.0 - self.date_weight) * self.category_loss(predictions[:,:-1], category_target)
        loss += self.date_weight * self.date_loss(predictions[:,-1:], date_range)
        return loss

