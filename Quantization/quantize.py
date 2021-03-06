"""
Methods for quantizing model.
"""
from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.nn as nn
import config

def static_quantize(model, inplace=True):
    '''
    model passed must be set to eval mode (model.eval()) for static 
    quantization logic to work
    '''
    torch.backends.quantized.engine = 'fbgemm'
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # model= torch.quantization.fuse_modules(model, [['conv', 'relu']])
    model=torch.quantization.prepare(model)
    model=torch.quantization.convert(model.cpu())
    return model     

def dynamic_quantize(model):
    '''
    model passed must be set to an instance of model (model()) 
    for dynamic quantize logic to work
    '''
    model = torch.quantization.quantize_dynamic(model, {nn.Sequential}, dtype=torch.qint8)
    return model

def qat_quantize_prepare(model, inplace=True):
    '''
    model passed must be set to train mode (model.train()) for quantization 
    aware training logic to work
    '''
    torch.backends.quantized.engine = 'fbgemm'
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.quantization.prepare_qat(model, inplace)

    model.eval()
    model = torch.quantization.convert(model)
    return model

def qat_quantize_eval(model):
    '''
    model passed must have been custom trained
    '''
    model.eval()
    model = torch.quantization.convert(model)
    return model







    