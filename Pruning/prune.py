"""
Methods for pruning model.
"""
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
from torch import nn
import logging
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class PruningModule(nn.Module):
    '''
    Module containing methods for pruning
    takes a pytorch model
    '''

    def prune_by_percentile(self, amount=5.0):
        '''
        method to prune specified modules in layer to be pruned with 
        percentile threshold
        '''
        alive_parameters = []
        for name, p in self.named_parameters():
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]
            alive_parameters.append(alive)

        all_alives = np.concatenate(alive_parameters)
        percentile_value = np.percentile(abs(all_alives), amount)
        logging.info(f'Pruning with threshold : {percentile_value}')

        for name, module in self.named_modules():
            if 'denselayer' in name:
                prune.ln_structured(module, name='weight', amount=percentile_value, n=2, dim=0)

    def l1_unstructured_pruning(self, percent=0.2):
        '''
        method to prune specified modules in layer using 
        l1 unstructured pruning
        '''
        logging.info(f'Pruning densenet layers using l1 unstructured pruning with threshold : {percent * 100}%')

        for name, module in self.named_modules():
            if 'denselayer' in name:
                prune.l1_unstructured(module, name='weight', amount=percent)
            if isinstance(module, nn.BatchNorm2d):
                prune.l1_unstructured(module, name='weight', amount=percent)

    def random_structured_pruning(self, amount=2, dim=1):
        '''
        method to prune specified modules in layer using 
        random unstructured pruning
        '''
        for name, module in self.named_modules():
            if 'denselayer' in name:
                prune.random_structured(module, name='weight', amount=amount, dim=dim)

    def global_unstructured_pruning(self, percent=0.3):
        '''
        method to prune specified modules in layer using 
        global unstructured pruning
        '''
        logging.info(f'Pruning using global unstructured pruning with threshold : {percent * 100}%')

        parameters_to_prune = (
            (self.model.avgpool2d, 'weight'),
            (self.model.conv2, 'weight'),
            (self.model.linear, 'weight'),
            (self.model.batchnorm2, 'weight'),
            (self.model.denselayer, 'weight')
        )
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=percent)


def global_unstructured_pruning(model, percent=0.7):
        '''
        method to prune specified modules in layer using 
        global unstructured pruning
        '''
        logging.info(f'Pruning using global unstructured pruning with threshold : {percent * 100}%')

        parameters_to_prune = (
            (model.features.denseblock1.denselayer1.norm1, 'weight'),
            (model.features.denseblock1.denselayer1.conv1, 'weight'),
            (model.features.denseblock1.denselayer1.norm2, 'weight'),
            (model.features.denseblock1.denselayer1.conv2, 'weight'),
            (model.features.denseblock1.denselayer2.norm1, 'weight'),
            (model.features.denseblock1.denselayer2.conv1, 'weight'),
            (model.features.denseblock1.denselayer2.norm2, 'weight'),
            (model.features.denseblock1.denselayer2.conv2, 'weight'),
            (model.features.denseblock1.denselayer3.norm1, 'weight'),
            (model.features.denseblock1.denselayer3.conv1, 'weight'),
            (model.features.denseblock1.denselayer3.norm2, 'weight'),
            (model.features.denseblock1.denselayer3.conv2, 'weight'),
            (model.features.denseblock1.denselayer4.norm1, 'weight'),
            (model.features.denseblock1.denselayer4.conv1, 'weight'),
            (model.features.denseblock1.denselayer4.norm2, 'weight'),
            (model.features.denseblock1.denselayer4.conv2, 'weight'),
            (model.features.denseblock1.denselayer5.norm1, 'weight'),
            (model.features.denseblock1.denselayer5.conv1, 'weight'),
            (model.features.denseblock1.denselayer5.norm2, 'weight'),
            (model.features.denseblock1.denselayer5.conv2, 'weight'),
            (model.features.denseblock1.denselayer6.norm1, 'weight'),
            (model.features.denseblock1.denselayer6.conv1, 'weight'),
            (model.features.denseblock1.denselayer6.norm2, 'weight'),
            (model.features.denseblock1.denselayer6.conv2, 'weight'),
            (model.features.denseblock2.denselayer1.norm1, 'weight'),
            (model.features.denseblock2.denselayer1.conv1, 'weight'),
            (model.features.denseblock2.denselayer1.norm2, 'weight'),
            (model.features.denseblock2.denselayer1.conv2, 'weight'),
            (model.features.denseblock2.denselayer2.norm1, 'weight'),
            (model.features.denseblock2.denselayer2.conv1, 'weight'),
            (model.features.denseblock2.denselayer2.norm2, 'weight'),
            (model.features.denseblock2.denselayer2.conv2, 'weight'),
            (model.features.denseblock2.denselayer3.norm1, 'weight'),
            (model.features.denseblock2.denselayer3.conv1, 'weight'),
            (model.features.denseblock2.denselayer3.norm2, 'weight'),
            (model.features.denseblock2.denselayer3.conv2, 'weight')
        )
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=percent)

        prune.remove(model.features.denseblock1.denselayer1.norm1, 'weight')
        prune.remove(model.features.denseblock1.denselayer1.conv1, 'weight')
        prune.remove(model.features.denseblock1.denselayer1.norm2, 'weight')
        prune.remove(model.features.denseblock1.denselayer1.conv2, 'weight')
        prune.remove(model.features.denseblock1.denselayer2.norm1, 'weight')
        prune.remove(model.features.denseblock1.denselayer2.conv1, 'weight')
        prune.remove(model.features.denseblock1.denselayer2.norm2, 'weight')
        prune.remove(model.features.denseblock1.denselayer2.conv2, 'weight')
        prune.remove(model.features.denseblock1.denselayer3.norm1, 'weight')
        prune.remove(model.features.denseblock1.denselayer3.conv1, 'weight')
        prune.remove(model.features.denseblock1.denselayer3.norm2, 'weight')
        prune.remove(model.features.denseblock1.denselayer3.conv2, 'weight')
        prune.remove(model.features.denseblock1.denselayer4.norm1, 'weight')
        prune.remove(model.features.denseblock1.denselayer4.conv1, 'weight')
        prune.remove(model.features.denseblock1.denselayer4.norm2, 'weight')
        prune.remove(model.features.denseblock1.denselayer4.conv2, 'weight')
        prune.remove(model.features.denseblock1.denselayer5.norm1, 'weight')
        prune.remove(model.features.denseblock1.denselayer5.conv1, 'weight')
        prune.remove(model.features.denseblock1.denselayer5.norm2, 'weight')
        prune.remove(model.features.denseblock1.denselayer5.conv2, 'weight')
        prune.remove(model.features.denseblock1.denselayer6.norm1, 'weight')
        prune.remove(model.features.denseblock1.denselayer6.conv1, 'weight')
        prune.remove(model.features.denseblock1.denselayer6.norm2, 'weight')
        prune.remove(model.features.denseblock1.denselayer6.conv2, 'weight')
        prune.remove(model.features.denseblock2.denselayer1.norm1, 'weight')
        prune.remove(model.features.denseblock2.denselayer1.conv1, 'weight')
        prune.remove(model.features.denseblock2.denselayer1.norm2, 'weight')
        prune.remove(model.features.denseblock2.denselayer1.conv2, 'weight')
        prune.remove(model.features.denseblock2.denselayer2.norm1, 'weight')
        prune.remove(model.features.denseblock2.denselayer2.conv1, 'weight')
        prune.remove(model.features.denseblock2.denselayer2.norm2, 'weight')
        prune.remove(model.features.denseblock2.denselayer2.conv2, 'weight')
        prune.remove(model.features.denseblock2.denselayer3.norm1, 'weight')
        prune.remove(model.features.denseblock2.denselayer3.conv1, 'weight')
        prune.remove(model.features.denseblock2.denselayer3.norm2, 'weight')
        prune.remove(model.features.denseblock2.denselayer3.conv2, 'weight')

        return model

def random_structured_pruning(model, amount=2, dim=1):
        '''
        method to prune specified modules in layer using 
        random unstructured pruning
        '''
        for name, module in model.named_modules():
        #     print(name)
        #     if 'denselayer' in name:
        #         print(name)
        #         print(module)
            prune.random_structured(module, name='weight', amount=amount, dim=dim)
        
        return model