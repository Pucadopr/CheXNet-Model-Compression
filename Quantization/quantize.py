from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.nn as nn

def static_quantize(model, inplace=True):

    torch.backends.quantized.engine = 'fbgemm'
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace)
    torch.quantization.convert(model.cpu(), inplace)
    
    return model

def dynamic_quantize(model):

    torch.quantization.quantize_dynamic(model, {nn.Sequential}, dtype=torch.qint8)
    return model

    