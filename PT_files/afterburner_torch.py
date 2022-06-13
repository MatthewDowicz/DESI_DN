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
                w_end):
    
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
    
    # Create 2 arrays of same size. full takes the denoised pixel values
    # and count keeps track of how many times an individual pixel has
    # inference ran on it
    full = np.empty((1, 1, 6000, 6000))
    count = np.empty((1, 1, 6000, 6000))
    
    # Get the noisy image data 
    noise_data = dataset[0]
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
        
        # We get the values that the denoised patch corresponds to in the
        # larger 'full' image and replaces the values of 'full' with the
        # denoised values. We also 'count' which pixels had inference ran
        # on them.
        full[:, :, h_start:h_end, w_start:w_end] += resid_img
        count[:, :, h_start:h_end, w_start:w_end] += 1
        
        # Deleting things to save memory.
        torch.cuda.empty_cache()
        del resid_img        
        
    return full, count

def full_img_pass(dataset,
                    model,
                    model_params,
                    samp_idx,
                    window_size=2000):
    
    """
    Full inference pass. Ie. this goes over every single pixel
    within the entire 6k by 6k image
    
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
    
    # Get size of the patch that we'll be running inference on.
    # Usually it will be 2000x2000. Then we see how many times our
    # patch fits within the full 6000x6000 image.
    inf_patch_size = window_size
    inf_patch_length = int(len(dataset[0][0][0]) / inf_patch_size)

    # We then create the indices of where one inference patch ends and
    # the other one begins and save those tuples together in a list for
    # use later.
    window_end_idx = []
    for k in range(inf_patch_length):
        window_end_idx.append(inf_patch_size*(k))
    window_end_idx.append(len(dataset[0][0][0])) # appends endpt ie. 6k

    # Full image pass
    # Instantiate 2 arrays of 6000x6000 for saving the denoised inference
    # patches in the correct location within the full 6k by 6k image, as 
    # well as counting which pixels have inference ran on them.
    full = np.empty((1, 1, 6000, 6000))
    count = np.empty((1, 1, 6000, 6000))

    # Loop through the patch indices, so as to create a loop that runs 
    # inference on non-overlapping regions of the full 6000x6000 image.
    # For a 2k by 2k patch we'd run this the model 9 times to cover the
    # FVC image.
    for j in range(len(window_end_idx)-1):
        for i in range(len(window_end_idx)-1):

            full_c1, count_c1 = grid_window(dataset=dataset,
                                            model=model,
                                            model_params=model_params,
                                            samp_idx=samp_idx,
                                            h_start=window_end_idx[i],
                                            h_end=window_end_idx[i+1],
                                            w_start=window_end_idx[j],
                                            w_end=window_end_idx[j+1])

            full += full_c1
            count += count_c1
            
            
    return full, count

def vertical_inf_pass(dataset,
                        model,
                        model_params,
                        samp_idx,
                        window_size=2000,
                        window_move_dist=1000):
    """
    Vertical inference pass. Ie. this doesn't cover the left & right
    most 1000 pixels (ie. should have a 1000 pixel wide empty column
    on left and right)
    
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
    window_size: int
        Width/height of the inference window that moves over the full 
        6k by 6k image.
        Defaults to 2000, which means that each inference call is over a
        2000x2000 sub-image of the full 6000x6000 FVC image.
    window_move_dist: int
        The distance the inference window moves between each calculation.
        Defaults to 1000, which is the distance between one inference
        calculation and the next. Ie. there will be an overlap of 1000 pixels.
        
    Returns:
    --------
    full_v: np.array
        Array of the models output over the specified window regions.
    count_v: np.array
        Array of 1's that keeps track of which pixels have had
        inferenced done upon them. This is so later on averaging can
        be done for pixels that had overlapping inference window
        calculations.
    """
    # Setting inference window size
    inf_patch_size = window_size
    # Calc. number of windows per row for 2k model it is 3
    inf_patch_length = int(len(dataset[0][0][0]) / inf_patch_size)
    # Setting how far each window should move
    window_move_distance = window_move_dist

    # Calc the indices where a window should end for a full row
    # ie. window_end_idx is [2000, 4000, 6000] for a 2k model
    window_end_idx = []
    for i in range(inf_patch_length):
        window_end_idx.append(inf_patch_size*(i+1))

    # Calc the start & indices for the cols that are going to used
    # for 2k model it is [1000, 3000, 5000] so there are cols of
    # 2 window length 
    direction_dep_window = list(np.array(window_end_idx) - window_move_distance)

    # Creating tuples for the start & end indices for the height
    # of the windows
    vert_pass_end_idxs = []
    for i in range(len(window_end_idx)):
        try:
            vert_pass_end_idxs.append((direction_dep_window[i], direction_dep_window[i+1]))
        except IndexError:
            pass

    full_vtot = np.empty((1, 1, 6000, 6000))
    count_vtot = np.empty((1, 1, 6000, 6000))
                        
    # Vertical pass
    for i in range(len(window_end_idx)):
        for j in range(len(vert_pass_end_idxs)):

            full_v, count_v = grid_window(dataset=dataset,
                                          model=model,
                                          model_params=model_params,
                                          samp_idx=samp_idx,
                                          h_start=0,
                                          h_end=window_end_idx[i],
                                          w_start=vert_pass_end_idxs[j][0],
                                          w_end=vert_pass_end_idxs[j][1])
        
            full_vtot += full_v
            count_vtot += count_v
    
    return full_vtot, count_vtot


def horizontal_inf_pass(dataset,
                        model,
                        model_params,
                        samp_idx,
                        window_size=2000,
                        window_move_dist=1000):
    """
    Horizontal inference pass. Ie. this doesn't cover the upper & lower
    most 1000 pixels (ie. should have a 1000 pixel high empty row
    on top and bottom)
    
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
    window_size: int
        Width/height of the inference window that moves over the full 
        6k by 6k image.
        Defaults to 2000, which means that each inference call is over a
        2000x2000 sub-image of the full 6000x6000 FVC image.
    window_move_dist: int
        The distance the inference window moves between each calculation.
        Defaults to 1000, which is the distance between one inference
        calculation and the next. Ie. there will be an overlap of 1000 pixels.
        
        
    Returns:
    --------
    full_htot: np.array
        Array of the models output over the specified window regions.
    count_htot: np.array
        Array of 1's that keeps track of which pixels have had
        inferenced done upon them. This is so later on averaging can
        be done for pixels that had overlapping inference window
        calculations.
    """
    # Setting inference window size
    inf_patch_size = window_size
    # Calc. number of windows per row
    # for 2k model it is 3
    inf_patch_length = int(len(dataset[0][0][0]) / inf_patch_size)
    # Setting how far each window should move
    window_move_distance = window_move_dist

    # Calc the indices where a window should end for a full row
    # ie. window_end_idx is [2000, 4000, 6000] for a 2k model
    # even end idxs
    window_end_idx = []
    for i in range(inf_patch_length):
        window_end_idx.append(inf_patch_size*(i+1))

    # Calc the start & indices for the cols that are going to used
    # for 2k model it is [1000, 3000, 5000] so there are cols of
    # 2 window length 
    direction_dep_window = list(np.array(window_end_idx) - window_move_distance)

    # Creating tuples for the start & end indices for the height
    # of the windows
    # odd end indices
    horiz_pass_end_idxs = []
    for i in range(len(window_end_idx)):
        try:
            horiz_pass_end_idxs.append((direction_dep_window[i], direction_dep_window[i+1]))
        except IndexError:
            pass

    full_htot = np.empty((1, 1, 6000, 6000))
    count_htot = np.empty((1, 1, 6000, 6000))

    for i in range(len(window_end_idx)):
        for j in range(len(horiz_pass_end_idxs)):
            
            full_h, count_h = grid_window(dataset=dataset,
                                          model=model,
                                          model_params=model_params,
                                          samp_idx=samp_idx,
                                          h_start=horiz_pass_end_idxs[j][0],
                                          h_end=horiz_pass_end_idxs[j][1],
                                          w_start=0,
                                          w_end=window_end_idx[i])
        
            full_htot += full_h
            count_htot += count_h
    
    return full_htot, count_htot

def afterburner(dataset,
                     model,
                     model_params,
                     samp_idx,
                     window_size=2000,
                     window_move_dist=1000):
    
    """
    Computes every inference pass. Ie. the horizontal (empty top & 
    bottom of inference image), the vertical (empty right & left
    column of inference image), and the full (entire image covered)
    pass. Once every pass is completed calculates the average pixel
    value for every pixel.
    
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
    window_size: int
        Width/height of the inference window that moves over the full 
        6k by 6k image.
        Defaults to 2000, which means that each inference call is over a
        2000x2000 sub-image of the full 6000x6000 FVC image.
    window_move_dist: int
        The distance the inference window moves between each calculation.
        Defaults to 1000, which is the distance between one inference
        calculation and the next. Ie. there will be an overlap of 1000 pixels.
       
        
    Returns:
    --------
    full_avg: np.array
        Averaged array of the 3 different full pixel arrays from the 
        different passes.
    count_v: np.array
        Array of 1's that keeps track of which pixels have had
        inferenced done upon them. This is so later on averaging can
        be done for pixels that had overlapping inference window
        calculations.
    """
    
   

    full_v, count_v = vertical_inf_pass(dataset=dataset,
                        model=model,
                        model_params=model_params,
                        samp_idx=samp_idx,
                        window_size=window_size,
                        window_move_dist=window_move_dist)


    
    full_h, count_h = horizontal_inf_pass(dataset=dataset,
                        model=model,
                        model_params=model_params,
                        samp_idx=samp_idx,
                        window_size=window_size,
                        window_move_dist=window_move_dist)
    
    
    full, count = full_img_pass(dataset=dataset,
                        model=model,
                        model_params=model_params,
                        samp_idx=samp_idx,
                        window_size=window_size)
    

    
    tot_full = full + full_v + full_h
    tot_count = count + count_v + count_h
    
    full_avg = tot_full / tot_count
    
    return full_avg