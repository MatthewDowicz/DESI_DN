# DnCNN Model for Denoising FVC Images

# NOTE: 
This repository works for me and is a good representation of my work. However, it won't work without the FVC image dataset files, which were created from running my notebook `00_FrontIlluminated.ipynb`. To access these files you'll need an account on [NERSC](https://www.nersc.gov/) and to be a member of [DESI](https://www.desi.lbl.gov/). All scripts and notebooks require Python 3.


- `PT_files` stands for PyTorch Files. This directory holds all the different PyTorch scripts necessary for 
    1. Preprocessing the raw data into usable datasets
    2. Creating PyTorch compatible Datasets & Data Generators
    3. Creating the DnCNN PyTorch model
    4. Creating the PyTorch afterburner that tiles over the full FVC image.

- `DnCNN_NP` stands for Denoising CNN Numpy. This directory holds the different numpy scripts necessary for running **just** the forward pass of DnCNN using the PyTorch models trained weights
    1. Creating the layers of the numpy model to create the index arrays
    2. Creating the layers that use the created index arrays 
    3. Creates the forward pass of DnCNN in numpy
    4. Creates the Numpy afterburner that tiles over the full FVC image.
    
# Project Purpose
The purpose of this project was to create a quick and automated algorithm that could denoise images of the fiber view camera (FVC) on the Dark Energy Spectroscopic Instrument (DESI). To understand the purpose one needs to understand some basics about DESI.

DESI is a spectroscopic instrument with 5,000 small independently moving robots. The benefit of the freedom of movement for the robots is to allow for 5,000 unique objects to be observed without the need to move the entire telescope to observe a single object. These robots needs to be ”parked” each night, so that they are in the predetermined location
the observing scripts expects them to be in. If the robots are not correctly parked this could lead to neighboring robots colliding with each other and damaging themselves or inhibiting neighboring robots from observing their assigned object. Due to these problems being massively important to minimize, so as to get the maximal amount of time on sky, Astronomers inspect the FVC images every day to ensure the robots are correctly positioned. The regions of the FVC images that are hardest to discern are in regions of low luminosity and high noise. Thus, we want a quick automated algorithm that removes noise to improve the inspection efficiency.
    
# Project Description
This repository contains my scripts and Jupyter Notebooks that 
- Trains a denoising CNN on images of the DESI FVC images 
- Denoises DESI FVC images in two different packages:
    - Pytorch for speed
    - Numpy for stability.
The .py scripts do most of the work, that said, certain notebooks do a some work namely: `00_Front_Illuminated.ipynb`, `01_Full_training_flow.ipynb`, `03_WB_sweep.ipynb`, `07_np_arr_creation.ipynb`, and `12_finalizing_scripts_work.ipynb`. The notebooks make use of certain .py scripts, but are the main engine due to their ease of testing. Most of the notebooks will eventually be put into scripts, but as of right now they are not.

## Most Important Scripts

### PT_files/new_afterburner_torch.py
This is the afterburner function used for PyTorch models trained on 2000x2000 sub-image patches of the FVC image. It does this by padding the full FVC image by 10 pixels on every side and uses patch sizes of 2010x2010. These patches overlap each other by 10 pixels and are then cropped, so that there artifacts induced by the overlapping region.

### DnCNN_NP/np_afterburner.py
This is the tiling afterburner function implemented in NumPy. It uses the weights trained by the PyTorch model and two auxiliary arrays to run the forward pass. It also pads the full FVC image and then crops it in a way to avoid inducing artifacts in overlapping patch regions.

<!-- ### PT_files/model.py

This houses multiple variants of DnCNN, but the most important/used model is DnCNN_B. This script just creates the model as well as allows for forward pass of the model to be conducted, ie. allows for images to be denoised. Even though DnCNN outputs the residual image the implementation already does the subtraction of the residual image from the input image to give us the denoised image. This was done for ease of use rather than mis-implemntation of the model.

## PT_files/preprocess_data.py
Given the raw data taken from the DESI Focal Plane Pipeline on NERSC preprocess the data into PyTorch compatible format, that is (N,C,H,W), as well as create training and test sets of user specified sizes.

### PT_files/Dataset.py
This creates the PyTorch compatible Dataset objects for our numpy saved data. 

### PT_files/afterburner_torch.py
This is the tiling afterburner function used for PyTorch models trained on 2000x2000 sub-image patches of the FVC images. This script does 3 unique types of passes, so as to account for potential discontinuities between inference patch regions.  -->

1. **Full Image Pass**: The model is shown 9 unique 2000x2000 image patches that do not overlap and runs inference on those patches and stitches together these 2000x2000 denoised patches into the larger 6000x6000 denoised FVC image.
2. **Vertical Image Pass**: The model is shown 6 2000x2000 image patches that overlap certain regions that **Full Image Pass** has already ran inference on. The 6 patches would be from `[0:6000,1000:3000]` and `[0:6000, 3000:5000]`
3. **Horizontal Image Pass**: The model is shown 6 other 2000x2000 image patches that overlap region regions that **Full Image Pass** has already ran inference on. The 6 patches would be from `[1000:3000,0:6000]`, and `[3000:5000, 0:6000]` if the model being used was trained for 2000x2000 images

## Most Important Jupyter Notebooks
Be warned, the notebooks are somewhat messy. They are quick implementations needed for quick visualizations. I use(d) the notebooks to generate plots, train the model quickly, and calculate metrics of the output of DnCNN.

### 01_Full_training_flow.ipynb
This notebook goes from the raw numpy arrays taken from NERSC and puts them into a correct format, then into PyTorch compatible dataset objects to be used in PyTorch Dataloaders. The model is then trained and the model weights are saved.

### 10_DnCNN_metrics.ipynb 
This notebook runs inference on a noisy FVC image by using `PT_files/afterburner_torch.py` then runs metric calculations on the denoised image with respect to the corresponding clean image. It also plots some visualizations of the output of the denoised image next to its corresponding clean/noisy image pair. 

### 12_finalizing_scripts_work.ipynb
This notebook calls the different model (numpy/pytorch) scripts to denoise noisy FVC images. This is the end result of the work/what the user on `fpoffline` should be using.