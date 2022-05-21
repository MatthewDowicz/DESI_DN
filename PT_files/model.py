import torch
from torch import nn
from typing import Any, Callable, Sequence, Optional

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



# Define model
class DnCNN(nn.Module):
    """
    Pytorch implementation of the Denoising CNN by Zhang et al. 2017
    in 'Beyond a Gaussian Denoiser: Residual Learning of Deep CNN
    for Image Denoising'.
    
    Model learns the residual (ie. noise) image of the data & then
    subtracts that residual image from the noisy image to obtain
    a denoised image. 
    """
    def __init__(self, num_layers=20, num_features=64):
        super(DnCNN, self).__init__()
        layers=[nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3,
                                                  padding=1),
                                       nn.BatchNorm2d(num_features),
                                       nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)
        
        self._initialize_weights()
        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        
    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        #return residual
        return y - residual
    
# Define model
class DnCNN_B(nn.Module):
    """
    Pytorch implementation of the Denoising CNN by Zhang et al. 2017
    in 'Beyond a Gaussian Denoiser: Residual Learning of Deep CNN
    for Image Denoising'.
    
    Model learns the residual (ie. noise) image of the data & then
    subtracts that residual image from the noisy image to obtain
    a denoised image. 
    """
    def __init__(self, num_layers=20, num_features=64):
        super(DnCNN_B, self).__init__()
        layers=[nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3,
                                                  padding=1),
                                       nn.BatchNorm2d(num_features),
                                       nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)
        
        self._initialize_weights()
        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        
    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        #return residual
        return y - residual
    
    
# Define model
class DnCNN_2k(nn.Module):
    """
    Pytorch implementation of the Denoising CNN by Zhang et al. 2017
    in 'Beyond a Gaussian Denoiser: Residual Learning of Deep CNN
    for Image Denoising'.
    
    Model learns the residual (ie. noise) image of the data & then
    subtracts that residual image from the noisy image to obtain
    a denoised image. 
    """
    def __init__(self, num_layers=10, num_features=32):
        super(DnCNN_2k, self).__init__()
        layers=[nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3,
                                                  padding=1),
                                       nn.BatchNorm2d(num_features),
                                       nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)
        
        self._initialize_weights()
        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        
    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        #return residual
        return y - residual