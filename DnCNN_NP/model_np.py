import numpy as np
import PT_files.save_load as sl
from DnCNN_NP.layers_np import relu, np_BatchNorm2d, np_Conv2d

import time 
from collections import OrderedDict
import pdb
import pathlib 
import os





def np_DnCNN(input_data, weights_dict, layer_list, im2col_mat):
    """
    Numpy version of the model architecture used for inference.
    
    Parameters:
    -----------
    input_data: nd.array
        Array containing the image (or sub-image) to be denoised
    weights_dict: OrderedDict
        Dictionary containing the trained weights for each layer
    layer_list: list
        List of the functions per layer. Due to the structure of the model
        having multiple transformations within one layer
        (ie. conv+batchnorm+relu = 1 layer) using this list of layer names
        allows for the correct order of transformations to be executed.
    im2col_mat: nd.array
        Array containing the `im2col` transformed image into a large 2d-matrix 
        to conduct the convolution operation as a dot product with the
        transformed weights matrix (ie. weight matrix also becomes
        large 2d-matrix).
        
    Returns:
    -------
    resid_img: nd.array
        The denoised image.
    """
    # First layer
    output = np_Conv2d(input_data=input_data,
                       weights_dict=weights_dict,
                       prefix=layer_list[0],
                       im2col_mat=im2col_mat,
                       col_prefix='start')
    output = relu(output)
    
    # Layer 2 - Layer 19
    for i in range(len(layer_list)-2):
    
        if layer_list[i+1].endswith('0.'):
            output = np_Conv2d(input_data=output,
                               weights_dict=weights_dict,
                               prefix=layer_list[i+1],
                               im2col_mat=im2col_mat,
                               col_prefix='mid')
            
        elif layer_list[i+1].endswith('1.'):
            
            output = np_BatchNorm2d(x=output, 
                                    weights_dict=weights_dict,
                                    prefix=layer_list[i+1])
            output = relu(output)
            
    # Layer 20 (last layer)
    output = np_Conv2d(input_data=output,
                       weights_dict=weights_dict,
                       prefix=layer_list[-1],
                       im2col_mat=im2col_mat,
                       col_prefix='last')
    
    resid_img = input_data - output
    
    return resid_img
