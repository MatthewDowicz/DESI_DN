import numpy as np
import PT_files.save_load as sl
from DnCNN_NP.layers_np import relu, np_BatchNorm2d, np_Conv2d
from DnCNN_NP.model_np import np_DnCNN

import time 
from collections import OrderedDict
import pdb
import pathlib 
import os



def grid_window(data,
                model,
                weights_dict,
                layer_list,
                im2col_mat,
                h_start,
                h_end,
                w_start,
                w_end):
    
    """
    Function to calculate a specified sized inference window.
    
    Parameters:
    -----------
    dataset: np.array
        Dataset of coupled noisy & clean images of full 6k images
    model: Numpy model
        np_DnCNN
    model_params: str
        Models parameters for the 2k image trained model
    samp_idx: int
        Sample index to select which of the test images to be used for 
        inference
    h_start: int
        The height starting index of the inference window.
        E.g. It would be the origin for the y-coord in a 2-D plot
    h_end: int
        The height ending index of the inference window.
        E.g. It would be the end of the y-axis for the y-coord 
        in a 2-D plot
    w_start: int
        The horizontal starting index of the inference window.
        E.g. It would be the origin for the x-coord in a 2-D plot
    w_end: int
        The horizontal ending index of the inference window.
        E.g. It would be the end of the x-axis for the x-coord 
        in a 2-D plot
   
        
    Returns:
    --------
    full: np.array
        Array of the models output over the window region.
    count: np.array
        Array of 1's that keeps track of which pixels have had
        inferenced done upon them. This is so later on averaging can
        be done for pixels that had overlapping inference window
        calculations.
    """
    
    full = np.empty((1, 1, 6000, 6000))
    count = np.empty((1, 1, 6000, 6000))
    
    # There might be a problem here if we're indexing too many indices if we only have a single image.
    noise_data = data[:, :, h_start:h_end, w_start:w_end]   

    # Load the model with corresponding weights, layer lists, and im2col matrices
    # and run inference (ie. denoise the image)
    denoised_img = np_DnCNN(input_data=noise_data, 
                     weights_dict=weights_dict,
                     layer_list=layer_list,
                     im2col_mat=im2col_mat)
        
    # Keep the denoised pixels together + the number of times
    # specific pixels had inference ran on them
    full[:, :, h_start:h_end, w_start:w_end] += denoised_img
    count[:, :, h_start:h_end, w_start:w_end] += 1
        
        
    return full, count


def np_full_img_pass(dataset,
                     model,
                     weights_dict,
                     layer_list,
                     im2col_mat,
                     window_size=2000):
    
    """
    Full inference pass. Ie. this goes over every single pixel
    within the entire 6k by 6k image
    
    Parameters:
    -----------
    dataset: np.array
        Dataset of coupled noisy & clean images of full 6k images
    model: Numpy model
        np_DnCNN
    weights_dict: Dict
        Dictionary of the weights for a 2k np_DnCNN model.
    window_size: int
        Width/height of the inference window that moves over the full 
        6k by 6k image. 
        Defaults to 2000, which means that each inference call is over a
        2000x2000 sub-image of the full 6000x6000 FVC image.
        
    Returns:
    --------
    full: np.array
        Array of the models output over the specified window regions.
    count: np.array
        Array of 1's that keeps track of which pixels have had
        inferenced done upon them. This is so later on averaging can
        be done for pixels that had overlapping inference window
        calculations.
    """
    
   
    inf_patch_size = window_size
    inf_patch_length = int(len(dataset[0][0][0]) / inf_patch_size)

    window_end_idx = []
    for k in range(inf_patch_length):
        window_end_idx.append(inf_patch_size*(k))
    window_end_idx.append(len(dataset[0][0][0])) # appends endpt ie. 6k

    # Full image pass
    full = np.zeros((1, 1, 6000, 6000))
    count = np.zeros((1, 1, 6000, 6000))

    for j in range(len(window_end_idx)-1):
        for i in range(len(window_end_idx)-1):

            full_c1, count_c1 = grid_window(data=dataset,
                                            model=model,
                                            weights_dict=weights_dict,
                                            layer_list=layer_list,
                                            im2col_mat=im2col_mat,
                                            h_start=window_end_idx[i],
                                            h_end=window_end_idx[i+1],
                                            w_start=window_end_idx[j],
                                            w_end=window_end_idx[j+1])

            full += full_c1
            count += count_c1
            print('Run finished.')
            
    return full, count