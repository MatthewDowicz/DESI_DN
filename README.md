# DnCNN Model for Denoising FVC Images

# NOTE: 
The values described below while describing `PT_files/afterburner_torch.py` is assuming you are feeding the model 2000x2000 patches. This is to have a clear example for describing the code. You can *probably* change the 

## Most Important Scripts

### PT_files/model.py

This houses multiple variants of DnCNN, but the most important/used model is DnCNN_B. This script just creates the model as well as allows for forward pass of the model to be conducted, ie. allows for images to be denoised. Even though DnCNN outputs the residual image the implementation already does the subtraction of the residual image from the input image to give us the denoised image. This was done for ease of use rather than mis-implemntation of the model.

## PT_files/preprocess_data.py
Given the raw data taken from the DESI Focal Plane Pipeline on NERSC preprocess the data into PyTorch compatible format, that is (N,C,H,W), as well as create training and test sets of user specified sizes.

### PT_files/Dataset.py
This creates the PyTorch compatible Dataset objects for our numpy saved data. 

### PT_files/afterburner_torch.py
This is the tiling afterburner function used for PyTorch models trained on 2000x2000 sub-image patches of the FVC images. This script does 3 unique types of passes, so as to account for potential discontinuities between inference patch regions. 

1. **Full Image Pass**: The model is shown 9 unique 2000x2000 image patches that do not overlap and runs inference on those patches and stitches together these 2000x2000 denoised patches into the larger 6000x6000 denoised FVC image.
2. **Vertical Image Pass**: The model is shown 6 2000x2000 image patches that overlap certain regions that **Full Image Pass** has already ran inference on. The 6 patches would be from `[0:6000,1000:3000]` and `[0:6000, 3000:5000]`
3. **Horizontal Image Pass**: The model is shown 6 other 2000x2000 image patches that overlap region regions that **Full Image Pass** has already ran inference on. The 6 patches would be from `[1000:3000,0:6000]`, and `[3000:5000, 0:6000]` if the model being used was trained for 2000x2000 images

## Most Important Jupyter Notebooks
Be warned, the notebooks are somewhat messy. They are quick implementations needed for quick visualizations. I use(d) the notebooks to generate plots, train the model quickly, and calculate metrics of the output of DnCNN.

### 01_Full_training_flow.ipynb
This notebook goes from the raw numpy arrays taken from NERSC and puts them into a correct format, then into PyTorch compatible dataset objects to be used in PyTorch Dataloaders. The model is then trained and the model weights are saved.

### 15_DnCNN_metrics.ipynb 
This notebook runs inference on a noisy FVC image by using `PT_files/afterburner_torch.py` then runs metric calculations on the denoised image with respect to the corresponding clean image. It also plots some visualizations of the output of the denoised image next to its corresponding clean/noisy image pair. 
