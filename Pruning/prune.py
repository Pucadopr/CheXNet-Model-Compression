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

class PruningModule(Module):
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
            if name.contains('denselayer'):
                module.prune(threshold=percentile_value)

    def l1_unstructured_pruning(self, percent=0.2):
        '''
        method to prune specified modules in layer using 
        l1 unstructured pruning
        '''
        logging.info(f'Pruning densenet layers using l1 unstructured pruning with threshold : {percent * 100}%')

        for name, module in self.named_modules():
            if name.contains('denselayer'):
                prune.l1_unstructured(module, name='weight', amount=percent)
            if isinstance(module, nn.BatchNorm2d):
                prune.l1_unstructured(module, name='weight', amount=percent)

    def random_structured_pruning(self, amount=2, dim=1):
        '''
        method to prune specified modules in layer using 
        random unstructured pruning
        '''
        for name, module in self.named_modules():
            if name.contains('denselayer'):
                prune.random_structured(module, name='weight', amount=amount, dim=dim)

    def global_unstructured_pruning(self, percent=0.3):
        '''
        method to prune specified modules in layer using 
        global unstructured pruning
        '''
        logging.info(f'Pruning using global unstructured pruning with threshold : {percent * 100}%')

        parameters_to_prune = (
            (self.avgpool2d, 'weight'),
            (self.Conv2d, 'weight'),
            (self.linear, 'weight'),
            (self.BatchNorm2d, 'weight'),
            (self.denselayer, 'weight')
        )
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=percent)
