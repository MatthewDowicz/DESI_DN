import numpy as np

from typing import Any, Callable, Sequence, Optional

import pathlib
import os
import pickle



def data_load(path, name):
    """ 
    Method to load the desired focal plane image data.
        Naming convention of raw data is:
    
        dataXXX-XXXX.npy:
        
            XXX - number of samples found in dataset
            XXXX - size of samples
            
        Naming convention for training/test data:
        
        sample_dataXXX_XXX.npy:
        
            XXX - number of samples found in dataset
            XXXX - size of samples
            
    Parameters:
    -----------
    path: str
        Path of directory holding the focal plane images
    name: str
        Name of particular focal plane image dataset that the user
        wants loaded
        
    Returns:
    --------
    np.array:
        The np.load'ed dataset of interest
            
    
    """
    
    return np.load(path + name)


def NERSC_load(name: str, Perlmutter=True):
    """ 
    Method to load the desired focal plane image data while working
    at NERSC. That means automatically knowing where the data is
    located.
            
    Parameters:
    -----------        
    name: str
        Name of particular focal plane image dataset that the user
        wants loaded
        
    Returns:
    --------
    np.array:
        The np.load'ed dataset of interest
        
        
     Naming convention of data is:
        ---------------------------------
    
        dataXXX-XXXX.npy:
        
            XXX - number of samples found in dataset
            XXXX - size of samples
    """
    
    if Perlmutter: 
        PATH = pathlib.Path(os.getenv('PSCRATCH'))
        DATA = PATH / 'DESI_dn' /'Data'
        assert DATA.exists()
        
        return np.load(DATA / name, allow_pickle=True)
        
    
    if not Perlmutter:
        PATH = pathlib.Path(os.getenv('SCRATCH'))
        DATA = PATH / 'DESI_dn' / 'Data'
        assert DATA.exists()
        
        return np.load(DATA / name, allow_pickle=True)
    
    else:
        print('You are not on NERSC?')


# def NERSC_save(name: str, data: np.array, Perlmutter=True):
#     """ 
#     Method to save the desired focal plane image data while working
#     at NERSC. That means automatically knowing where the data is
#     supposed to be located.
            
#     Parameters:
#     -----------        
#     name: str
#         Name of particular focal plane image dataset that the user
#         wants saved
        
#     data: np.array
#         The np.array pixel data of the focal plane image dataset
        
#     Returns:
#     --------
#     None
        
        
#     Naming convention of data is:
#     ---------------------------------
#         dataXXX-XXXX.npy:
        
#             XXX - number of samples found in dataset
#             XXXX - size of samples
#     """
#     if Perlmutter: 
#         PATH = pathlib.Path(os.getenv('PSCRATCH'))
#         DATA_PATH = PATH / 'DESI_dn' /'Data'
#         assert DATA_PATH.exists()

#         return np.save(DATA_PATH / name, data)

    
#     if not Perlmutter:
#         PATH = pathlib.Path(os.getenv('SCRATCH'))
#         DATA_PATH = PATH / 'DESI_dn' / 'Data'
#         assert DATA_PATH.exists()

#         return np.save(DATA_PATH/ name, data)
    
#     else:
#         print('You are not on NERSC?')
def NERSC_save(name: str, data, Perlmutter=True):
    """ 
    Method to save the desired focal plane image data while working
    at NERSC. That means automatically knowing where the data is
    supposed to be located.
            
    Parameters:
    -----------        
    name: str
        Name of particular focal plane image dataset that the user
        wants saved
        
    data: np.array
        The np.array pixel data of the focal plane image dataset
        
    Returns:
    --------
    None
        
        
    Naming convention of data is:
    ---------------------------------
        dataXXX-XXXX.npy:
        
            XXX - number of samples found in dataset
            XXXX - size of samples
    """
    if Perlmutter: 
        PATH = pathlib.Path(os.getenv('PSCRATCH'))
        DATA_PATH = PATH / 'DESI_dn' /'Data'
        assert DATA_PATH.exists()
        
        file_path = DATA_PATH / name
        with file_path.open('wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if not Perlmutter:
        PATH = pathlib.Path(os.getenv('SCRATCH'))
        DATA_PATH = PATH / 'DESI_dn' / 'Data'
        assert DATA_PATH.exists()

        file_path = DATA_PATH / name
        with file_path.open('wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    else:
        print('You are not on NERSC?')



# def load(path: str, name: str , NERSC=True, Perlmutter=True):
#     """ 
#     Method to load the desired focal plane image data.
#         Naming convention of data is:
#         ---------------------------------
    
#         dataXXX-XXXX.npy:
        
#             XXX - number of samples found in dataset
#             XXXX - size of samples
            
#     Parameters:
#     -----------
#     path: pathlib.PosixPath
#         Path of directory holding the focal plane images.
#         NOTE: If on NERSC should be either $SCRATCH$ or $pscratch$
#               due to the sizes of the focal plane images
        
#     name: str
#         Name of particular focal plane image dataset that the user
#         wants loaded
        
#     Returns:
#     --------
#     np.array:
#         The np.load'ed dataset of interest
#     """
    
#     # User is on NERSC
#     if NERSC:
#         # This is used during the period of Cori & Perlmutter usage
#         # Should be deprecated as Perlmutter becomes fully operational
#         if Perlmutter:
#             perl_data_path = pathlib.Path(os.getenv('PSCRATCH'))
#             assert perl_data_path.exists()
            
#             dataset = np.load(
#     # data_path = pathlib.Path(os.getenv(path))
#     # assert data_path.exists()
    
#     return np.load(path + name)

