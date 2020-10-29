"""
Utility methods for training and pruning model.
"""
import torch
import os
import shutil
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def check_path_exists(path):
            
    if not os.path.isdir(path):
        raise Exception("image path must be a directory")

def check_file_exists(file):

    if not os.path.isfile(file):
        raise Exception("csv file passed does not exist")


def save_checkpoint(state, is_best, filepath):
    # we should specify a filepath instead.
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))

    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))
        # torch.save(state, 'checkpoint/{}_best.pth.tar'.format(filename))


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size= os.path.getsize("temp.p")/1e6
    print('Size (MB):', size)
    os.remove('temp.p')
    
    return size

