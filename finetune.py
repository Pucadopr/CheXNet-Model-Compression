"""
Method to finetune pretrained models using compression techniques
"""
from pruning import prune as pr
from quantization import quantize as quant
from test import validate
import utils
from data import get_nih_data_paths
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import random_split
import logging
import torchxrayvision as xrv
from utils import print_size_of_model


def finetune_pretrained_model(model, compress_type, batch_size, log_interval, percent=0.2):
    '''
    finetune a pretrained model using compression techniques defined.
    '''

    logging.info(f"Using pretrained {model} model for compression")

    model = xrv.models.DenseNet(weights=model)
    print_size_of_model(model)

    image_path, x_ray_path, bbox_path = get_nih_data_paths()

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    d_nih = xrv.datasets.NIH_Dataset(imgpath=image_path, transform=transform)
    # print(d_nih)
    
    criterion = nn.BCELoss()
    d_nih_len = len(d_nih)

    no_train_data = int(0.8*d_nih_len)
    no_test_data = int(0.2*d_nih_len)+1

    logging.info(f"Using a total of {no_test_data} for model validation")

    train_data, test_data = torch.utils.data.random_split(d_nih, [no_train_data, no_test_data])
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size= batch_size, shuffle=True, num_workers=8)

    accuracy= validate(test_dataloader, model, criterion, "cpu", log_interval)

    logging.info(f"Initial accuracy of model is {accuracy} %")
    
    if compress_type=='prune':
        prune = pr.PruningModule(model)
        prune.l1_unstructured_pruning(percent)

        print_size_of_model(model)

        accuracy= validate(test_dataloader, model, criterion, "cpu", log_interval)

        logging.info(f"Accuracy after pruning model is {accuracy} %")   
    
    if compress_type=='quantize':
        model= model.eval()
        model = quant.static_quantize(model)
        
        print_size_of_model(model)

        accuracy= validate(test_dataloader, model, criterion, "cpu", log_interval)

        logging.info(f"Accuracy after quantization is {accuracy} %")
