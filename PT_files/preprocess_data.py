import numpy as np

from typing import Any, Callable, Sequence, Optional

import pathlib
import os

from .save_load import NERSC_save, NERSC_load


def create_data_sets(data: np.array,
                     train_size: int,
                     test_size: int,
                     training_set_name: str,
                     test_set_name: str) -> np.array:
    """
    Method that creates training & test sets of user specific sizes
    from the raw focal plane image data in the form of:
    
    dataset = (Number of samples, number of channels, height, width)
    
    NOTE: This is done so the data can be used in a Pytorch
          Dataloader object. This file should not be used in the production
          version of the Pytorch Denoiser used in fpoffline.
          
    Parameters:
    -----------
    data : np.array
        A 4-D array containing - (sample type, number of samples,
                                  sample_height, sample_width)
            
            data[idx]: Noisy data samples (idx=0) & 
                       clean data samples (idx=1)       
    train_size: int
        Size of the training set.
    test_size: int
        Size of the test set.
    training_set_name: str
        Name that will be saved for this training set.
        Convention:
        -----------
        test_dataXXX-XXXX.npy
            XXX - number of samples
            XXXX - size of a individual sample
    test_set-name: str
        Name that will be saved for this test set.
        Convention:
        -----------
        test_dataXXX-XXXX.npy
            XXX - number of samples
            XXXX - size of a individual sample
        
    Returns:
    --------
    
    
    """
    # Load in the noise data and
    # add a new dim to act as channel dimension for training in pytorch
    D_noise = data[0]
    D_noise = D_noise[:, np.newaxis, :, :]
    
    D_noise_train = D_noise[:train_size]
    D_noise_test = D_noise[train_size:]
    assert len(D_noise_test) == test_size, "Test set not correct size!"
    
    
    D_clean = data[1]
    D_clean = D_clean[:, np.newaxis, :, :]
    
    D_clean_train = D_clean[:train_size]
    D_clean_test = D_clean[train_size:]
    assert len(D_clean_test) == test_size, "Test set not correct size!"
    
    training_data = np.stack((D_noise_train, D_clean_train))
    test_data = np.stack((D_noise_test, D_clean_test))
    
    # return training_data, test_data
    NERSC_save(str(training_set_name), training_data)
    NERSC_save(str(test_set_name), test_data)
    
    
    
    
def create_full_img_data_sets(data: np.array,
                     train_size: int,
                     test_size: int,
                     training_set_name: str,
                     test_set_name: str) -> np.array:
    """
    Method that creates training & test sets of user specific sizes
    from the raw focal plane image data in the form of:
    
    dataset = (Number of samples, number of channels, height, width)
    
    NOTE: This is done so the data can be used in a Pytorch
          Dataloader object.
          
    Parameters:
    -----------
    data : np.array
        A 4-D array containing - (sample type, number of samples,
                                  sample_height, sample_width)
            
            data[idx]: Noisy data samples (idx=0) & 
                       clean data samples (idx=1)    
    train_size: int
        Size of the training set.
    test_size: int
        Size of the test set.
    training_set_name: str
        Name that will be saved for this training set.
        Convention:
        -----------
        test_dataXXX-XXXX.npy
            XXX - number of samples
            XXXX - size of a individual sample
    test_set-name: str
        Name that will be saved for this test set.
        Convention:
        -----------
        test_dataXXX-XXXX.npy
            XXX - number of samples
            XXXX - size of a individual sample
        
    Returns:
    --------
    
    
    """
    # Load in the noise data and
    # add a new dim to act as channel dimension for training in pytorch
    D_noise = data[0]
    D_noise = D_noise[:, np.newaxis, :, :]
    
    D_noise_train = D_noise[:train_size]
    D_noise_test = D_noise[train_size:]
    assert len(D_noise_test) == test_size, "Test set not correct size!"
    
    
    D_clean = data[1]
    D_clean = D_clean[:, np.newaxis, :, :]
    
    D_clean_train = D_clean[:train_size]
    D_clean_test = D_clean[train_size:]
    assert len(D_clean_test) == test_size, "Test set not correct size!"
    
    training_data = np.stack((D_noise_train, D_clean_train))
    test_data = np.stack((D_noise_test, D_clean_test))
    
    # return training_data, test_data
    NERSC_save(str(training_set_name), training_data)
    NERSC_save(str(test_set_name), test_data)