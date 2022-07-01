import numpy as np
import torch
import PT_files.save_load as sl


from DnCNN_NP.layers_np import relu, np_BatchNorm2d, np_Conv2d
from DnCNN_NP.model_np import np_DnCNN

import time 
from collections import OrderedDict
import pdb
import pathlib 
import os

def numpy_burner(data,
                 model=np_DnCNN,
                 weights_dict='2k_model_bs64_e800_ps50_Adam.pth',
                 layer_list='2k_NP_layers_list.pkl',
                 im2col_mat='im2col_2k_indices.pkl',
                 filepath='/global/cfs/cdirs/desi/engineering/focalplane/endofnight/denoise/',
                 patch_size=2000,
                 padding=10):
    
    """
    Numpy FVC denoiser. 
    
    It runs via a sliding window method, where it denioses patchs of ~2kx2k 
    pixels. Doing it this way cuts down on memory and speeds up inference.
    
    
    Parameters:
    -----------
    data: np.array
        Input FVC image.
    model: Numpy model
        np_DnCNN
    weights_dict: Dict
        Dictionary of the weights for a 2k np_DnCNN model.
        Default is '2k_model_bs64_e800_ps50_Adam.pth'
    layer_list: str
        Name of the file that holds the names for each layer. Used in calling
        correct weights (for conv & bn layers) & index matrices to conduct 
        the `im2col` algorithm to quickly compute convolution layers.
        Default is '2k_NP_layers_list.pkl'.
    im2col_mat: str
        Name of the file that contains the index arrays for the numpy conv
        layers. These are needed for the `im2col` algorithm that computes
        the convolutions in an efficient and fast way.
        Default is 'im2col_2k_indices.pkl'.
    filepath: str
        Path to the directory housing the weights & indice arrays for the 
        numpy model.
        Default is '/global/cfs/cdirs/desi/engineering/focalplane/endofnight/denoise/.
    patch_size: int
        Width/height of the inference window that moves over the full 
        6k by 6k image. 
        Defaults to 2000.
    padding: int
        How much to pad the FVC image and patch image. Padding it allows
        for slight overlap between denoised patchs. This allows for no
        artifacts to come about, thus having a pure denoised image.
        Default is 10.
        
    Returns:
    --------
    full: np.array
        Full denoised FVC image.
    """
    
    # Load the saved im2col matrices of indices & im2col layer lists  
    # Create the path to the directory that houses the files.
    PATH = pathlib.Path(filepath)
    im2col_mat_path = PATH / im2col_mat
    assert im2col_mat_path.exists()
    layer_list_path = PATH / layer_list
    assert layer_list_path.exists()
    
    # Load the files from their specific paths
    loaded_im2col_mat = np.load(im2col_mat_path, allow_pickle=True)
    loaded_layer_list = np.load(layer_list_path, allow_pickle=True)
    
    # Get correct path to the model weights on shared diredctory
    # This is an interim shared directory & will update once David sends me
    # the new path that will be used in fpoffline.
    model_path = PATH / weights_dict
    assert model_path.exists()
    weights = torch.load(str(model_path))
    

    # Reshape the image to be in the correct format to be used in the model.  
    noisy = np.reshape(data, (1, 1, 6000, 6000))
    # Pad the FVC image for overlapping patchs
    # Doing this eliminates artifacts that show up at the edges of the
    # individual patchs.
    noisy = np.pad(noisy, ((0,0), (0, 0), (padding, padding), (padding, padding)))

    # Get how many patchs fit within our FVC image.
    # E.g. patchs_per_dim == 3 if patch_size == 2000
    # We're indexing on the H axis of the img ie. data[0][0] == 6000
    patchs_per_dim = int(len(data) / patch_size)
    
    # Create the indices of where one patch ends and the other one begins.
    # Save those together in a list for later use. Expected output will
    # be [0, 2000, 4000, 6000] if using patch_size=2000.
    window_end_idx = []
    for k in range(patchs_per_dim):
        window_end_idx.append(patch_size*(k))
    window_end_idx.append(len(data)) # appends endpt ie. 6k
    
    # Full image pass
    full = np.zeros((1, 1, 6000, 6000))

    for j in range(len(window_end_idx)-1):
        for i in range(len(window_end_idx)-1):

            # This gets the 2020x2020 patch we want to run denoise.
            # Adding the padding gets the H/W to be 2020 instead of 2000
            noise_data = noisy[:, :, 
                               window_end_idx[i]:window_end_idx[i+1]+(padding*2),
                               window_end_idx[j]:window_end_idx[j+1]+(padding*2)]
            
            # Run the model on this patch of the FVC image
            denoised_patch =  model(input_data=noise_data, 
                                       weights_dict=weights,
                                       layer_list=loaded_layer_list,
                                       im2col_mat=loaded_im2col_mat)
            
            # Crop the 10x10 border of the denoised_patch, so that when
            # we stitch it together there's no artifacts between neighboring
            # patchs.
            denoised_patch = denoised_patch[:, :, 10:-10, 10:-10]
            
            # Add the denoised patch to the empty array of FVC size
            full[:, :, window_end_idx[i]:window_end_idx[i+1],
                 window_end_idx[j]:window_end_idx[j+1]] += denoised_patch
            
    return full[0][0]
