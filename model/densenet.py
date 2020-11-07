"""
DenseNet 121 Model definition.
"""
from operator import contains
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import torchvision
import torch.nn.functional as F
from torchvision.models.densenet import DenseNet
from torchvision.models.mobilenet import ConvBNReLU


class DenseNet121(nn.Module):
    '''
    DenseNet121 model initialized with number of classes, if to use a pretrained
    network and if model to be defined is for quantization aware training
    '''

    def __init__(self, num_classes, is_pretrained, quantized=False):
        super(DenseNet121, self).__init__()
        self.quantize = quantized
        if self.quantize:
            self.quant = QuantStub()
        self.densenet121 = torchvision.models.densenet121(pretrained=is_pretrained)
        self.features = self.densenet121.features
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )
        if quantized:
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        features = self.features(x)
        features = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        out = self.classifier(out)
        if self.quantize:
            out = self.dequant(out)
        return out

    def fuse_model(self):
        for name, module in self.named_modules():
            if type(module) == DenseNet:
                torch.quantization.fuse_modules(module, ['conv0', 'norm0', 'relu0'], inplace=True)
                
            if name.contains('denselayer'):
                        torch.quantization.fuse_modules(module, [['norm1', 'relu1', 'conv1'],['norm2', 'relu2', 'conv2']], inplace=True)


