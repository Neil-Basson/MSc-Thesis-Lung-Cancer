import os
import numpy as np
import glob

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms

import albumentations as albu
from albumentations.pytorch import ToTensorV2

class LIDCDataset(Dataset):
    def __init__(self, IMAGES_PATHS, MASK_PATHS, Augmentation = False):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        MASKS_PATHS: list of masks paths ['./Masks/0001_01_masks.npy','./Masks/0001_02_masks.npy']
        """
        self.image_paths = IMAGES_PATHS
        self.mask_paths= MASK_PATHS
        self.augmentation = Augmentation

        
        self.augmentation_transformation =  albu.Compose([
            albu.ElasticTransform(alpha=1.1, sigma=5.0, p=0.15),
            albu.HorizontalFlip(p=0.15),
            ToTensorV2()
        ])
        self.transformation = transforms.Compose([transforms.ToTensor()])
    def transform(self, image, mask):
        # Apply augmentation and transformation
        if self.augmentation:
            # Reshape to work with albumentations
            image = image.reshape(512,512,1)
            mask = mask.reshape(512,512,1)
            mask = mask.astype('uint8')
            augmented = self.augmentation_transformation(image=image,mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # Reshape to work with NN 
            mask = mask.reshape([1,512,512])
        # Apply transformation
        else:
            image = self.transformation(image)
            mask = self.transformation(mask)

        image, mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image,mask

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])
        image,mask = self.transform(image,mask)
        return image,mask

    def __len__(self):
        return len(self.image_paths)