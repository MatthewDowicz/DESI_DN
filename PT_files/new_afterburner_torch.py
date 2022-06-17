import numpy as np
import torch
from torch import nn
import pathlib
device = "cuda" if torch.cuda.is_available() else "cpu"

def grid_window(dataset,
                model,
                model_params,
                samp_idx,
                h_start,
                h_end,
                w_start,
                w_end,
                padding):
    
    """
    Function to calculate a specified sized inference window.
    
    Parameters:
    -----------
    dataset: np.array
        Dataset of coupled noisy & clean images of full 6k images
    model: Pytorch model
        DnCNN
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
    padding: int
        How much to pad the FVC 6kx6k image. This is so we can 
        take larger patch sizes and thus have overlapping patchs
        that reduces the creation of artifacts within the denoised
        image.
   
        
    Returns:
    --------
    full: np.array
        Array of the models output over the window region.
    """
    
    # Create 2 arrays of same size. full takes the denoised pixel values
    # and count keeps track of how many times an individual pixel has
    # inference ran on it
    # full = np.empty((1, 1, 6000, 6000))
    # count = np.empty((1, 1, 6000, 6000))
    
    # Get the noisy image data 
    noise_data = dataset
    noise_data = np.pad(noise_data, ((0,0), (0, 0), (padding, padding), (padding, padding)))
    params_name = model_params
    
    # Get the correct patht to the moodel weights
    current_dir = pathlib.Path().resolve()
    model_params_path = current_dir / 'Model_params'
    assert model_params_path.exists()
    model_path = model_params_path / params_name
    
    # Instantiate the model, put it onto the GPU, load the weights
    # of the trained model, and then set the model into evaluation mode
    # for inference.
    model = model()
    model.to(device)
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()

    # Turn off gradient tracking so as to not update
    # the model weights
    with torch.no_grad():
                
        # Delete any remaining memory, turn the sub_image patch numpy array
        # into a torch tensor, so as to be compatible with the model, and
        # then put the data onto the GPU so as to be in the same place as 
        # the model.
        torch.cuda.empty_cache()
        test_noise = torch.as_tensor(noise_data[samp_idx:samp_idx+1, :, h_start:h_end, w_start:w_end])
        test_noise = test_noise.to(device)

        # Run the model on the noisy images, then detach the output from
        # the GPU and put it onto CPU while making it into a numpy array.
        # Delete 'output' so as to save memory
        output = model(test_noise)
        resid_img = output.detach().cpu().numpy()
        del output
        
        # Same as above more or less
        test_noise.detach().cpu()
        torch.cuda.empty_cache()
        del test_noise
        
        # Deleting things to save memory.
        torch.cuda.empty_cache()
        
    return resid_img


def full_pass_torch(data, model, model_params, patch_size=2000, padding=10):
    """
    Function that uses a sliding window to run inference over the full FVC
    image. There is some overlap (~10 pixels) depending on the location of
    the patch.
    
    
    Parameters:
    -----------
    data: np.array
        Array of the noisy FVC exposure.
    model: DnCNN_B
        The denoising CNN model to be used.
    model_params: OrderedDict
        Pickled OrderedDict of the trained model weights.
    patch_size: int
        Size of the inference window.
        Defaults to 2000.
    padding: int
        How much to pad the input image to allow for artifact free 
        stitching of denoised patchs.
        Defaults to 10.
    """
    
    # Reshape the image to be in the correct format to be used in model.
    noisy = np.reshape(data, (1, 1, 6000, 6000))
    # Pad the full image at the end of the width and height of the image.
    # Meaning there will be an extra 10 pixels to the RHS and bottom of the
    # image resulting in FVC_img.shape = (1, 1, 6010, 6010)
    # noisy = np.pad(noisy, ((0,0), (0,0), (padding,padding), (padding,padding)))
    
    # Get how many patchs fit within our FVC image.
    # E.g. patchs_per_dim == 3 if patch_size == 2000
    patchs_per_dim = int(len(data) / patch_size)
    
    # Create the indices of where one patch ends and the other one begins.
    # Save those together in a list for later use. Expected output will
    # be [0, 2000, 4000, 6000] if using patch_size=2000.
    window_end_idx = []
    for k in range(patchs_per_dim):
        window_end_idx.append(patch_size*(k))
    window_end_idx.append(len(data)) # appends endpt. ie. 6k
    
    # Instantiate an array of 6000x6000 for saving the denoised inference
    # patches in the correct location within the full 6k by 6k image.
    full = np.zeros((1, 1, 6000, 6000))
    
    # Loop through the patch indices to create a loop that runs inference
    # on regions of the full 6000x6000 image. For a 2k by 2k patch we'd run
    # this 9 times to cover the entire FVC image.
    for j in range(len(window_end_idx)-1):
        for i in range(len(window_end_idx)-1):
            
            denoised_patch = grid_window(dataset=noisy,
                                        model=model,
                                        model_params=str(model_params),
                                        samp_idx=0,
                                        h_start=window_end_idx[i],
                                        h_end=window_end_idx[i+1]+(padding*2), 
                                        w_start=window_end_idx[j],
                                        w_end=window_end_idx[j+1]+(padding*2),
                                        padding=padding)
                                        # the reason for padding*2 is b/c
                                        # padding is for only one side of
                                        # the img, but we need to pad both.
                                        # so the *2 accounts for both sides
            
            denoised_patch = denoised_patch[:, :, 10:-10, 10:-10]
            
            full[:, :, window_end_idx[i]:window_end_idx[i+1],
                 window_end_idx[j]:window_end_idx[j+1]] += denoised_patch


    return full