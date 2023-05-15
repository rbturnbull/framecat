import torch
from fastai.metrics import accuracy as fa_accuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from torch import Tensor
from fastai.metrics import AccumMetric, ActivationType


DATE_MEAN = 1141.028124917435
DATE_STD = 173.93942833016163



def accuracy(predictions, *targets):
    return fa_accuracy(predictions[:,:2], targets[0])
    # target = targets[0].cpu()
    # predictions = torch.argmax(predictions[:,:2], dim=1).cpu()
    # correct = (predictions == target)
    # return correct.float().mean()


def f1_score(predictions, *targets):
    return f1(targets[0].cpu(), torch.argmax(predictions[:,:2], dim=1).cpu(), average="macro")


def precision_score(predictions, *targets):
    return precision(targets[0].cpu(), torch.argmax(predictions[:,:2], dim=1).cpu(), average="macro")


def recall_score(predictions, *targets):
    return recall(targets[0].cpu(), torch.argmax(predictions[:,:2], dim=1).cpu(), average="macro")


def date_accuracy(predictions, category_target: Tensor, date_range:Tensor):
    date_predictions = predictions[:,2:]
    correct = (date_range[:,0] <= date_predictions) & (date_predictions <= date_range[:,1])
    correct |= torch.abs( date_range.mean(dim=1) - date_predictions ) <= 50.0/DATE_STD

    return correct.float().mean()


class CategoryAccumMetric(AccumMetric):
    def accumulate(self, learn):
        pred = learn.pred[:,:2]
        y = learn.y
        if isinstance(y, tuple):
            y = y[0]

        if self.activation in [ActivationType.Softmax, ActivationType.BinarySoftmax]:
            pred = F.softmax(pred, dim=self.dim_argmax)
            if self.activation == ActivationType.BinarySoftmax: pred = pred[:, -1]
        elif self.activation == ActivationType.Sigmoid: pred = torch.sigmoid(pred)
        elif self.dim_argmax: pred = pred.argmax(dim=self.dim_argmax)
        if self.thresh:  pred = (pred >= self.thresh)
        self.accum_values(pred,y,learn)


def F1Score():
    return CategoryAccumMetric(f1_score, flatten=False)


def PrecisionScore():
    return CategoryAccumMetric(precision_score, flatten=False)


def RecallScore():
    return CategoryAccumMetric(recall_score, flatten=False)

