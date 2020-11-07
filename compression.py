"""
File to compress model using pruning or quantization techniques
"""
from pruning.prune import PruningModule
import quantization.quantize as qaunt
import utils
import argparse
from train import train
import test
import os
from data import get_nih_data_paths
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from model.densenet import DenseNet121
from dataset_loader import NIHDatasetLoader
from utils import print_size_of_model


parser = argparse.ArgumentParser(description='CheXNet Model compression for low edged devices')

parser.add_argument('--model', default='nih', choices=['kaggle', 'nih', 'pc', 'chex'],
                    help='model name for compression')
parser.add_argument('--compress-type', default='prune', choices=['prune', 'quantize'],
                    help='type of compression method to use')
parser.add_argument('--compress-method', default='dynamic', choices=['static', 'dynamic'],
                    help='type of compression method to use')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--weight-decay', type=int, default=0.00001, metavar='N',
                    help='weight decay for train (default: 0.00001)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")


args = parser.parse_args()

image_path, x_ray_path, bbox_path = get_nih_data_paths()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
dataset = NIHDatasetLoader(image_path, x_ray_path, bbox_path, transform)

train_dataset = []
test_dataset = []

for i, data in enumerate(dataset):
    train_size = int(0.5 * len(dataset))
    test_size = len(data) - train_size
    torch.manual_seed(args.seed)
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)

no_of_labels = 14
model = DenseNet121(no_of_labels)

if args.compress_type =='quantize':
    model = DenseNet121(14, False, True)
    model.fuse_model()

print_size_of_model(model)
model = model.cuda()

criterion = nn.BCELoss()
optimizer = optim.SGD(filter(
            lambda p: p.requires_grad,
            model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)

model =train(train_dataloader, model, criterion, optimizer, args.epochs, 10)






   