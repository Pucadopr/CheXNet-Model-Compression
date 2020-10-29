from torch import nn
import torchvision
import torch.nn.functional as F


class DenseNet121(nn.Module):
    
    def __init__(self, num_classes, is_pretrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=is_pretrained)
        self.features = self.densenet121.features
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        out = self.classifier(out)
        return out, features

    ## define methods to remove last layer or add to __init__
    ## define methods to load a weight and other relevant ones
