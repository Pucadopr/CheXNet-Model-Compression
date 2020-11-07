"""
File to compress model using pruning or quantization techniques
"""
from pruning.prune import PruningModule
import quantization.quantize 
import utils
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model.densenet import DenseNet121


