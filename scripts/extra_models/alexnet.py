'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn

import torch
import torch.nn.functional as F
from models._scda import SCDA
from models._crow import CroW


__all__ = ['alexnet']


class AlexNet(nn.Module):
    
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        self.name = "AlexNet"
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def obtain_features_gap(self, x):
        _features_module = torch.nn.Sequential(*list(self.features.modules())[1:-1])
        
        x = _features_module(x)
        x = F.avg_pool2d(x, x.size(-2), x.size(-1))
        x = x.view(x.shape[0], -1)
        return x
    
    def obtain_features_gap_all(self, x):
        out_all = None
    
        for i, module in enumerate(self.features.modules()):
            if i == 0:
                continue
                           
            if isinstance(module, nn.MaxPool2d):
                out = nn.AdaptiveAvgPool2d((1, 1))(x)
                out = out.view(out.shape[0], -1)
                out_all = torch.cat((out_all, out), dim=1) if out_all is not None else out
                
            x = module(x)
                
        return out_all

    def obtain_features_gmp(self, x):
        _features_module = torch.nn.Sequential(*list(self.features.modules())[1:-1])
        
        x = _features_module(x)
        x = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.shape[0], -1)
        return x

    def obtain_features_scda(self, x):
        scda = SCDA()
        _features_module = torch.nn.Sequential(*list(self.features.modules())[1:-1])
        
        x = _features_module(x)
        return scda(x)
        

    def obtain_features_crow(self, x):
        crow = CroW()
        _features_module = torch.nn.Sequential(*list(self.features.modules())[1:-1])
        
        x = _features_module(x)
        return crow(x)

def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
