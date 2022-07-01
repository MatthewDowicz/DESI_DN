import torch
from torch.utils.data import Dataset
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

    
class Img_Dataset(Dataset):
    def __init__(self, data_set, patch_size, width, height, seed=1234):
        """
        Parameters:
        -----------
        data: np.ndarray
            Array that contains image/label pairs ie. corrupted image/clean image.
            Shape = (P, N, C, H, W):
                P = corrupted/uncorrupted image pair 
                N = number of samples
                C = number of channels
                H = image height
                W = image width
        patch_size: int
            Size of randomly chosen image patch the model uses for training
        width: int
            Width of the chosen sample.
            NOTE: It's a parameter because you can input a larger image and choose
                  to look at only portions of said image for more training samples.
        height: int
            Height of the chosen sample.
        seed: int 
            Randomized seed used for the random slicing used to create the image patch.
        """
        self.data_set = data_set
        self.patch_size = patch_size
        self.width = width
        self.height = height
        self.seed = seed

    def __len__(self):
        return len(self.data_set[0])
    
    def __shape__(self):
        return self.data_set.shape

    def __getitem__(self, idx):
        """
        Function that returns the PyTorch Dataloader compatible dataset.
        
        Parameters:
        -----------
        idx: var
            Variable used in PyTorch Dataloader to be able to sample from the dataset
            to create minibatches of the data for us automatically.
        """
        # Loading the dataset and then slicing the image/label pairs 
        # ie. corrupted/uncorrupted images. 
        # Note the use of the idx in the image/label variables. This allows the
        # PyTorch Dataloader to get all the important data info eg. (N, C, H, W)
        data = self.data_set
        image = data[0, idx]
        label = data[1, idx]
        
        # Setting the patch size and the randomized seed for the image patch
        patch_size = self.patch_size
        seed = self.seed
        rng = np.random.RandomState(seed)

        img_width = self.width
        img_height = self.height
        
        #randomly crop patch from training set
        x1 = rng.randint(img_width - patch_size)
        y1 = rng.randint(img_height - patch_size)
        S = (slice(y1, y1 + patch_size), slice(x1, x1 + patch_size))
        
        # create new arrays for training patchs
        image_patch = image[0][S]
        label_patch = label[0][S]
        
        # add new axis to act as channel dimension that is necessary
        # for use with pytorch models/layers
        image_patch = image_patch[np.newaxis, :, :]
        label_patch = label_patch[np.newaxis, :, :]

        
        # Turning our image/label to a PyTorch Tensor with dtype = float 
        # and then putting it onto the GPU for faster training/inference
        image = torch.from_numpy(image_patch).float().to(device)
        label = torch.from_numpy(label_patch).float().to(device)
            
        return image, label
    
    
    
# class Large_Img_Dataset(Dataset):
#     def __init__(self, data_set, num_patchs, patch_size, width, height, seed=1234):
#         self.data_set = data_set
#         self.num_patchs = num_patchs
#         self.patch_size = patch_size
#         self.width = width
#         self.height = height
#         self.seed = seed

#     def __len__(self):
#         return len(self.data_set[0])
    
#     def __shape__(self):
#         return self.data_set.shape

#     def __getitem__(self, idx):
#         data = self.data_set
#         image = data[0, idx]
#         label = data[1, idx]
        
#         patch_size = self.patch_size
#         num_patchs = self.num_patchs
#         seed = self.seed
#         rng = np.random.RandomState(seed)

       
#         img_width = self.width
#         img_height = self.height
        
#         img_patch_list = []
#         label_patch_list = []
#         for i in range(num_patchs):
        
#             #randomly crop patch from training set
#             x1 = rng.randint(img_width - patch_size)
#             y1 = rng.randint(img_height - patch_size)
#             S = (slice(y1, y1 + patch_size), slice(x1, x1 + patch_size))

#             # create new arrays for training patchs
#             # append each patch_image to an empty list & convert to an
#             # np.array
#             image_patch = image[0][S]
#             img_patch_list.append(image_patch)
#             final_image = np.asarray(img_patch_list)
            
            
#             label_patch = label[0][S]
#             label_patch_list.append(label_patch)
#             final_label = np.asarray(label_patch_list)

#         # add new axis to act as channel dimension that is necessary
#         # for use with pytorch models/layers
#         # image_patch = final_image[np.newaxis, :, :]
#         # label_patch = final_label[np.newaxis, :, :]
#         image_patch = final_image
#         label_patch = final_label

#         image = torch.from_numpy(image_patch).float().cuda(device)
#         label = torch.from_numpy(label_patch).float().cuda(device)
            
#         return image, label