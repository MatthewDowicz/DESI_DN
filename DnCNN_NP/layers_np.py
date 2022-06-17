# This file is for Numpy version of layers where the index matrices
# have already been saved.
# See 'layers.py' if you want the functions that create the saved
# index matrices.
import numpy as np
import time 



# Activations
relu = lambda x: np.maximum(0, x)

# Layers
def np_BatchNorm2d(x, weights_dict, prefix, epsilon=1e-5):
    """
    Computes the batch normalized version of the input.
    
    This function implements a BatchNorm2d from PyTorch. A caveat to
    remember is that this implementation is equivalent to nn.BatchNorm2d
    in `model.eval()` mode. Batch normalization renormalizes the input 
    to the layer to a more parsable data range.
    
    Parameters:
    -----------
    x: numpy.ndarray
        Input image data.
    mean: numpy.ndarray
        Running mean of the dataset, computed during training.
    var: numpy.ndarray
        Running variance of the dataset, computed during training.
    beta: numpy.ndarray
        Offset value added to the normalized output.
        (These are the biases from the model parameter dictionary).
    gamma: numpy.ndarray
        Scale value to rescale the normalzied output.
        (These are the weights from the model parameter dictionary).
    epsilon: float
        Small constant for numerical stability. 
        Default = 1e-5.
        
    Returns:
    --------
    numpy.ndarray
        Output of the batch normalization.
        
    Notes:
    ------
    The operation implemented in this function is:
    
    .. math:: \\frac{\gamma (x - \mu)}{\sigma + \epsilon} + \\beta
    
    where :math:`\mu` is the running mean of the dataset and :math:`\sigma` is
    the running variance of the dataset, both of which are computed during
    training.
    
    For more details and documentation on the PyTorch BatchNorm2d function
    that this function mimics can be found at 
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """

    gamma = weights_dict[str(prefix) + 'weight'].detach().cpu().numpy().reshape(-1, 1, 1)
    beta = weights_dict[str(prefix) + 'bias'].detach().cpu().numpy().reshape(-1, 1, 1)
    mean = weights_dict[str(prefix) + 'running_mean'].detach().cpu().numpy().reshape(-1, 1, 1).reshape(-1, 1, 1)
    var = weights_dict[str(prefix) + 'running_var'].detach().cpu().numpy().reshape(-1, 1, 1)
    
    output = ((x - mean) / np.sqrt(var + epsilon)) * gamma + beta
    
    return output



# Functions for implementing Conv2d

def get_indices(input_data, weights_dict, prefix, stride=1, padding=1):
    # Get input size
    
    # Checking to see if a single sample or a batch of samples is given.
    # If batch take the batch_size, in_channels, H, and W
    # If single sample is given reshape so the values above can be calculated
    if len(input_data.shape) == 4:
    
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)
        
    elif len(input_data.shape) == 3:
        
        input_data = input_data.reshape((1, 1, 2000 , 2000))
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)
        
    # Load the weights and biases needed for a convolution
    # then take off gpu memory, move to CPU memory,
    # and lastly transform to numpy
    weight = weights_dict[str(prefix) + 'weight']
    weight = weight.detach().cpu().numpy()
    
    bias = weights_dict[str(prefix) + 'bias']
    bias = bias.detach().cpu().numpy()
    
    # Calculate the kernel size and output channels from
    # the loaded weights from above
    kernel_size = weight[0][0].shape
    output_channels = len(weight)
    
    # Calculations for the output H and W dimensions.
    height_out = ((height + (2*padding) - (kernel_size[0] - 1) - 1) / stride) + 1
    height_out = int(height_out)
    width_out = ((width + (2*padding) - (kernel_size[1] - 1) - 1) / stride) + 1
    width_out = int(width_out)
    
    
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(kernel_size[0]), kernel_size[1])
    # Duplicate for the other channels.
    level1 = np.tile(level1, input_channels)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(height_out), width_out)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    
    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(kernel_size[1]), kernel_size[0])
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, input_channels)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(width_out), height_out)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    
    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(input_channels), kernel_size[0] * kernel_size[1]).reshape(-1, 1)
    
    return i, j, d

def im2col(input_data, im2col_mat, col_prefix, stride=1, padding=1):
    """
        Transforms our input image into a matrix.

        Parameters:
        -----------
        input_data: nd.array
            The input image(s)
        weights_dict: OrderedDict
            Dictionary containing the PyTorch trained weights for every 
            layer of the model
        prefix: str
            The prefix that picks out the specific layer's weights to be used
            E.g. prefix='layers.0.0.' would be the first layers convolutional
            weights and bias's

        Returns:
        --------
        cols: output matrix.
    """
    if len(input_data.shape) == 4:
    
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)
        
    elif len(input_data.shape) == 3:
        
        input_data = input_data.reshape((1, 1, 2000 , 2000))
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)

    # Padding
    input_padded = np.pad(input_data, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
    # Multi-dimensional arrays indexing.
    idx = im2col_mat[str(col_prefix)]
    cols2 = input_padded.reshape(-1)[idx]  
    
    return cols2

def np_Conv2d(input_data, weights_dict, prefix, im2col_mat, col_prefix):
    """
        Performs a forward convolution.

        Parameters:
        - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
        Returns:
        - out: previous layer convolved.
    """
    
    if len(input_data.shape) == 4:
    
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)
        
    elif len(input_data.shape) == 3:
        
        # Change this code
        input_data = input_data.reshape((1, 1, 2020 , 2020))
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)


    output_channels = len(weights_dict[str(prefix) + 'weight']) # num_of_filters
    height_out = int((height + 2 * 1 - 3)/ 1) + 1
    width_out = int((width + 2 * 1 - 3)/ 1) + 1

    
    X_col = im2col(input_data=input_data, im2col_mat=im2col_mat, col_prefix=str(col_prefix))
    w_col = weights_dict[str(prefix) + 'weight'].detach().cpu().numpy().reshape((output_channels, -1))
    b_col = weights_dict[str(prefix) + 'bias'].detach().cpu().numpy().reshape(-1, 1)
    # Perform matrix multiplication.
    out = w_col @ X_col + b_col
    # Reshape back matrix to image.
    out = np.array(np.hsplit(out, batch_size)).reshape((batch_size, output_channels, height_out, width_out))

    return out