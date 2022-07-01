import torch
from torch import nn
from typing import Any, Callable, Sequence, Optional

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



# Define model
class DnCNN(nn.Module):
    """
    PyTorch implementation of DnCNN (https://arxiv.org/pdf/1608.03981.pdf).
    Model learns the noise of the input data and outputs the learned noise.
    The denoised image = input_img - learned_noise.
    """
    def __init__(self, num_layers=20, num_features=64):
        """
        Initialization of the model.
        """
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
    PyTorch implementation of DnCNN (https://arxiv.org/pdf/1608.03981.pdf).
    Model learns the noise of the input data and outputs the learned noise.
    The denoised image = input_img - learned_noise.
    """
    def __init__(self, num_layers=20, num_features=64):
        """
        Initialization of the model. 
        """
        super(DnCNN_B, self).__init__()
        
        # Create the first layer of the model. This is the yellow layer in 
        # Fig 1. of the paper. This takes a grayscale image and outputs 64 
        # maps. This does not reduce the dimensionality of the input image.
        layers=[nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))]
        
        # This loop creates the intermediate layers of the model (ie. not 
        # the first or last, but everything in between). These layers 
        # correspond to the blue layers in Fig 1. of the paper.
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3,
                                                  padding=1),
                                       nn.BatchNorm2d(num_features),
                                       nn.ReLU(inplace=True)))
        # This is the last layer of the model where we go from our many
        # (64) feature maps back to 1, so as to make sure the output 
        # is grayscale and allows for the subtraction of the output 
        # to the input to get the denoised image of interest
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        # Put all the layers into one model so the output of one layer
        # automatically goes into the next layer below it.
        self.layers = nn.Sequential(*layers)
        
        # Initialize the models weights
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
        # Get the output of the model
        residual = self.layers(y)
        # Return the residual image (ie. the denoised image)
        # Due to the fact that the model learns the noise of the input image
        # and not the deniosed image.
        return y - residual
    