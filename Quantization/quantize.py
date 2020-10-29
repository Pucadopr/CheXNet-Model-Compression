from torch.quantization import QuantStub, DeQuantStub
import torch

def quantize(model, inplace=True):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace)
    torch.quantization.convert(model.cpu(), inplace)

    return model