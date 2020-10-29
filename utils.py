"""
Utility methods for training and pruning model.
"""
import torch
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


def check_path_exists(path):
            
    if not os.path.isdir(path):
        raise Exception("image path must be a directory")

def check_file_exists(file):

    if not os.path.isfile(file):
        raise Exception("csv file passed does not exist")

def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, 'checkpoint/{}_best.pth.tar'.format(filename))

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size= os.path.getsize("temp.p")/1e6
    print('Size (MB):', size)
    os.remove('temp.p')
    
    return size
