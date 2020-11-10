"""
Method to finetune pretrained models using compression techniques
"""
from pruning.prune import PruningModule
import quantization.quantize as quant
from test import validate
import utils
from data import get_nih_data_paths
import torch
import torch.nn as nn
from torch.utils.data import random_split
import logging
import torchxrayvision as xrv
from utils import print_size_of_model


def finetune_pretrained_model(model, compress_type, batch_size, log_interval, percent=0.2 ):
    '''
    finetune a pretrained model using compression techniques defined.
    '''
    logging.info(f"Using pretrained {model} model for compression")
    model = xrv.models.DenseNet(model)
    print_size_of_model(model)

    image_path, x_ray_path, bbox_path = get_nih_data_paths()
    d_nih = xrv.datasets.NIH_Dataset(imgpath=image_path)
    criterion = nn.BCELoss()
    test_dataloader = torch.utils.data.DataLoader(d_nih, batch_size= batch_size, shuffle=True, num_workers=8)

    accuracy= validate(test_dataloader, model, criterion, "cuda", log_interval)

    logging.info(f"Initial accuracy of model is {accuracy} %")
    
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