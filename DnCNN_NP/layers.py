import numpy as np
from scipy import signal
import time 

#ORIGINAL NUMPY LAYERS. USED IN GETTING THE INTERMEDIATE 
#INDEX MATRICES AS WELL AS THE FULL IM2COL MATRICES. 
# These are NOT used in the actual model to denoise the FVC images.

# Activations
relu = lambda x: np.maximum(0, x)


# Required layers for DnCNN model
# Will have to update these due to the PyTorch weight dictionaries
# having dictionary key names.



# def np_Conv2d(input_data, weights_dict, stride=1, padding="same", dilation=1):
#     """
#     Numpy implementation of the PyTorch Conv2d layer that uses the 
#     learned PyTorch weights in the model.
    
#     Parameters:
#     -----------
#     input_data: nd.array
#         Input data of shape '(batch_size, in_channels, height, width)'
#     weights_dict: OrderedDict
#         weights_dict['weight']: torch.Tensor
#             Weights tensor of shape '(out_channels, in_channels, kernel_size[0], kernel_size[1])'
#         weights_dict['bias']: torch.Tensor
#             Bias tensor of shapee '(out_channels)'
#     stride: int, optional
#         The number of entries by which the filter is moved at each step.
#         Defaults to 1
#     padding: str, optional
#         What padding strategy to use for this conv layer. Defaults to "same",
#         which pads the layer such that the output has the same height and width
#         as the input when the stride = 1. Specifically makes output of
#         scipy.correlate2d have same shape as in1. An alternative option is "valid",
#         which means no padding is done and the output has smaller dimensions
#         than the input.
#     dilation: int, optional
#         Spacing between kernel elements.
#         Defaults to 1.
     
        
#     Returns:
#     --------
#     output: nd.array
#         Array output of the convolution step with shape
#         `(batch_size, out_channels, out_height, out_width)`.
    
#     """
    
#     # Watch the NN numpy version on youtube 
    
#     batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)
#     kernel_size = weights_dict['weight'][0][0].shape
#     output_channels = len(weights_dict['weight'])
    
#     # Convert string padding into numerical padding
#     # Using strings allow for one variable to account for padding & mode (see signal.correlated2d)
#     mode = padding
#     if mode == "same":
#         padding = 1
#     elif mode == "valid":
#         padding = 0
    
#     height_out = ((height + (2*padding) - dilation * (kernel_size[0] - 1) - 1) / stride) + 1
#     height_out = int(height_out)
#     width_out = ((width + (2*padding) - dilation * (kernel_size[1] - 1) - 1) / stride) + 1
#     width_out = int(width_out)

#     output = np.empty((batch_size, output_channels, height_out, width_out))
    
#     for i in range(batch_size):
#         for j in range(output_channels):
#             output[i, j, :, :] = weights_dict['bias'][j] + signal.correlate2d(input_data[i][0], weights_dict['weight'][j][0], mode=mode)
               
#     return output

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
    
    # Calculations for the output H and W dimensions
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

def im2col(input_data, weights_dict, prefix, stride=1, padding=1):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    
    if len(input_data.shape) == 4:
    
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)
        
    elif len(input_data.shape) == 3:
        
        input_data = input_data.reshape((1, 1, 2000 , 2000))
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)

    # Padding
    input_padded = np.pad(input_data, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
    i, j, d = get_indices(input_data=input_data, weights_dict=weights_dict, prefix=prefix)
    # Multi-dimensional arrays indexing.
    cols = input_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def np_Conv2d(input_data, weights_dict, prefix):
    """
        Performs a forward convolution.

        Parameters:
        - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
        Returns:
        - out: previous layer convolved.
    """
    
    conv_start = time.perf_counter()
    if len(input_data.shape) == 4:
    
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)
        
    elif len(input_data.shape) == 3:
        
        input_data = input_data.reshape((1, 1, 2000 , 2000))
        batch_size, input_channels, height, width = input_data.shape # (N, Cin, Hin, Win)


    output_channels = len(weights_dict[str(prefix) + 'weight']) # num_of_filters
    height_out = int((height + 2 * 1 - 3)/ 1) + 1
    width_out = int((width + 2 * 1 - 3)/ 1) + 1

    X_col = im2col(input_data=input_data, weights_dict=weights_dict, prefix=prefix)
    w_col = weights_dict[str(prefix) + 'weight'].detach().cpu().numpy().reshape((output_channels, -1))
    b_col = weights_dict[str(prefix) + 'bias'].detach().cpu().numpy().reshape(-1, 1)
    # Perform matrix multiplication.
    out = w_col @ X_col + b_col
    # Reshape back matrix to image.
    out = np.array(np.hsplit(out, batch_size)).reshape((batch_size, output_channels, height_out, width_out))
    
    conv_end = time.perf_counter()
    print('Conv takes', conv_end-conv_start, 'seconds')
    return out

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
    batch_start = time.perf_counter()

    gamma = weights_dict[str(prefix) + 'weight'].detach().cpu().numpy().reshape(-1, 1, 1)
    beta = weights_dict[str(prefix) + 'bias'].detach().cpu().numpy().reshape(-1, 1, 1)
    mean = weights_dict[str(prefix) + 'running_mean'].detach().cpu().numpy().reshape(-1, 1, 1).reshape(-1, 1, 1)
    var = weights_dict[str(prefix) + 'running_var'].detach().cpu().numpy().reshape(-1, 1, 1)
    
    output = ((x - mean) / np.sqrt(var + epsilon)) * gamma + beta
    
    batch_end = time.perf_counter()
    print('Batch takes', batch_end-batch_start, 'seconds')
    
    return output








#----------------------------------------------------------------------#


# FOR LOOP THAT CREATES THE MODEL. NEED TO CREATE IT INTO A FUNCTION.
# SAVED HERE FOR REFERENCE AND FOR FUTURE USE.
# THIS CAN BE FOUND IN THE '11_Testing_im2col_speed.ipynb`


# from collections import OrderedDict

# # Replace the last part of the key that describes what layer it is
# # part of and replaces it with empty space
# layers_list = [x.replace('weight', '').replace('bias', '').replace('running_mean', '').replace('running_var', '').replace('num_batches_tracked', '') for x in weights.keys()]
# # Convert this list which has duplicated elements due to removing
# # identifying elements ie. for the first conv layer we had
# # layers.0.0.weight & layers.0.0.bias, but now after removing them we
# # have layers.0.0 & layers.0.0
# # The code below deletes the duplicated elements
# layers_list = list(OrderedDict.fromkeys(layers_list))



# # 1st layer
# model_start = time.perf_counter()
# output = np_Conv2d(input_data=samp,
#                    weights_dict=weights,
#                    prefix=layers_list[0])

# output = relu(output)


# for i in range(len(layers_list)-2):
    
#     if layers_list[i+1].endswith('0.'):
        
#         # conv_start = time.perf_counter()
#         output = np_Conv2d(input_data=output,
#                            weights_dict=weights,
#                            prefix=layers_list[i+1])
#         conv_end = time.perf_counter()
#         # print('Conv Layer', conv_end-conv_start, 'seconds')
        
#     elif layers_list[i+1].endswith('1.'):
        
#         # batch_start = time.perf_counter()
#         output = np_BatchNorm2d(x=output,
#                                 weights_dict=weights,
#                                 prefix=layers_list[i+1])
#         output = relu(output)
#         batch_end = time.perf_counter()
#         # print('Batch Layer', batch_end-batch_start, 'seconds')


# output = np_Conv2d(input_data=output,
#                    weights_dict=weights,
#                    prefix=layers_list[-1])

# resid_img = samp - output

# model_end= time.perf_counter()
# print('Total time for 2k by 2k numpy inference takes', model_end-model_start, 'seconds')