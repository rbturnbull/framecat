import torch
from fastai.metrics import accuracy as fa_accuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from torch import Tensor


DATE_MEAN = 1141.028124917435
DATE_STD = 173.93942833016163


def accuracy(predictions, *targets):
    return fa_accuracy(predictions[:,:-1], targets[0])


def f1_score(predictions, *targets):
    return f1(targets[0].cpu(), torch.argmax(predictions[:,:-1], dim=1).cpu(), average="macro")


def precision_score(predictions, *targets):
    return precision(targets[0].cpu(), torch.argmax(predictions[:,:-1], dim=1).cpu(), average="macro")


def recall_score(predictions, *targets):
    return recall(targets[0].cpu(), torch.argmax(predictions[:,:-1], dim=1).cpu(), average="macro")


def date_accuracy(predictions, category_target: Tensor, date_range:Tensor):
    date_predictions = predictions[:,-1:]
    correct = (date_range[:,0] <= date_predictions) & (date_predictions <= date_range[:,1])
    correct |= torch.abs( date_range.mean(dim=1) - date_predictions ) <= 50.0/DATE_STD

    return correct.float().mean()