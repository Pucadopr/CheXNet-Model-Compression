"""
File to preprocess and train model using pruning or quantization techniques
"""
from pruning.prune import PruningModule
from quantization import quantize as quant
from test import validate
import utils
from train import train
from data import get_nih_data_paths
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import logging
from torchvision import datasets, transforms
from model.densenet import DenseNet121
from dataset_loader import NIHDatasetLoader
from utils import print_size_of_model


def preprocess_model(compress_type, batch_size, seed, lr, weight_decay, epochs, percent=0.2, log_interval=10):
    '''
    Method to preprocess, train and compress model using compresstype
    specified. also takes in other training parameters
    '''

    logging.info(f"downloading and preparing nih dataset for training")
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
        torch.manual_seed(seed)
        train_dataset, test_dataset = random_split(data, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size= batch_size, shuffle=True, num_workers=8)

    no_of_labels = 18

    # model = model.cuda()

    model = DenseNet121(no_of_labels)

    if compress_type =='quantize':
        model = DenseNet121(no_of_labels, False, True)
        model.fuse_model()
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(filter(
                lambda p: p.requires_grad,
                model.parameters()),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay)

    logging.info(f"commencing model training")

    if compress_type =='quantize':

        model =train(train_dataloader, model, criterion, optimizer, epochs, log_interval, True)
    else:
        model =train(train_dataloader, model, criterion, optimizer, epochs, log_interval)
    
    accuracy= validate(test_dataloader, model, criterion, "cuda", log_interval)

    logging.info(f"Accuracy of model after initial training is {accuracy} %")

    print_size_of_model(model)

    if compress_type=='prune':
        prune = PruningModule(model)
        prune.l1_unstructured_pruning(percent)

        print_size_of_model(model)

        accuracy= validate(test_dataloader, model, criterion, "cuda", log_interval)

        logging.info(f"Accuracy after pruning model is {accuracy} %")

    if compress_type=='quantize':
        model= model.eval()
        model = quant.static_quantize(model)
        
        print_size_of_model(model)

        accuracy= validate(test_dataloader, model, criterion, "cuda", log_interval)

        logging.info(f"Accuracy after quantization is {accuracy} %")

        ## is there retraining after quantizing, add more argument parses 