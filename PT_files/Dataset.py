import torch
from torch.utils.data import Dataset
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"


class Img_Dataset(Dataset):
    def __init__(self, data_set, patch_size, width, height, seed=1234):
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
        data = self.data_set
        image = data[0, idx]
        label = data[1, idx]
        
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
        
        image = torch.from_numpy(image_patch).float().cuda(device)
        label = torch.from_numpy(label_patch).float().cuda(device)
            
        return image, label
    
    
    
class Large_Img_Dataset(Dataset):
    def __init__(self, data_set, num_patchs, patch_size, width, height, seed=1234):
        self.data_set = data_set
        self.num_patchs = num_patchs
        self.patch_size = patch_size
        self.width = width
        self.height = height
        self.seed = seed

    def __len__(self):
        return len(self.data_set[0])
    
    def __shape__(self):
        return self.data_set.shape

    def __getitem__(self, idx):
        data = self.data_set
        image = data[0, idx]
        label = data[1, idx]
        
        patch_size = self.patch_size
        num_patchs = self.num_patchs
        seed = self.seed
        rng = np.random.RandomState(seed)

       
        img_width = self.width
        img_height = self.height
        
        img_patch_list = []
        label_patch_list = []
        for i in range(num_patchs):
        
            #randomly crop patch from training set
            x1 = rng.randint(img_width - patch_size)
            y1 = rng.randint(img_height - patch_size)
            S = (slice(y1, y1 + patch_size), slice(x1, x1 + patch_size))

            # create new arrays for training patchs
            # append each patch_image to an empty list & convert to an
            # np.array
            image_patch = image[0][S]
            img_patch_list.append(image_patch)
            final_image = np.asarray(img_patch_list)
            
            
            label_patch = label[0][S]
            label_patch_list.append(label_patch)
            final_label = np.asarray(label_patch_list)

        # add new axis to act as channel dimension that is necessary
        # for use with pytorch models/layers
        # image_patch = final_image[np.newaxis, :, :]
        # label_patch = final_label[np.newaxis, :, :]
        image_patch = final_image
        label_patch = final_label

        image = torch.from_numpy(image_patch).float().cuda(device)
        label = torch.from_numpy(label_patch).float().cuda(device)
            
        return image, label