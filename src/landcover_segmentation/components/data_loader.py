import os
os.environ['http_proxy'] = 'http://proxy1.bgc-jena.mpg.de:3128' 
os.environ['https_proxy'] = 'http://proxy1.bgc-jena.mpg.de:3128'

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import cv2
from PIL import Image

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from landcover_segmentation.logging import logger
from landcover_segmentation.entity import DataLoaderConfig

class DataLoaderSegmentation(torch.utils.data.Dataset):
    def __init__(
        self, 
        imgs_path,
        masks_path,
        transform=None
    ):
        super().__init__()
        self.imgs_path = imgs_path
        self.imgs_list = os.listdir(imgs_path)

        self.masks_path = masks_path
        self.masks_list = os.listdir(masks_path)

        self.transform = transform

    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_path, self.imgs_list[idx])
        mask_path = os.path.join(self.masks_path, self.masks_list[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
                
        return image, mask
    
    def get_image_paths(self):
        return [os.path.join(self.imgs_path, img) for img in self.imgs_list]


class SegDataLoader:
    def __init__(
        self,
        config: DataLoaderConfig
    ):
        self.config = config
        self.preprocess_input = get_preprocessing_fn(self.config.BACKBONE, pretrained=self.config.pretrained)

    def add_single_img_processing(self, img, mask):
        num_class = self.config.n_classes
        
        img = img.permute(0, 2, 3, 1)

        scaler = MinMaxScaler()
        img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        img = self.preprocess_input(img)  # Preprocess based on the pretrained backbone...
        
        # Convert mask to one-hot encoding
        mask = F.one_hot(mask.to(torch.int64), num_class)  # Convert to one-hot
        
        return img, mask

    def TrainGenerator(self, data_type: str):
        transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

        if data_type=='train':
            dataset = DataLoaderSegmentation(
                imgs_path=os.path.join(self.config.preprocessed_data_path, 'train_images', data_type),
                masks_path=os.path.join(self.config.preprocessed_data_path, 'train_masks', data_type),
                transform=transform
            )

        elif data_type=='test':
            dataset = DataLoaderSegmentation(
                imgs_path=os.path.join(self.config.preprocessed_data_path, 'test_images', data_type),
                masks_path=os.path.join(self.config.preprocessed_data_path, 'test_masks', data_type),
                transform=transform
            )
            
        elif data_type=='val':
            dataset = DataLoaderSegmentation(
                imgs_path=os.path.join(self.config.preprocessed_data_path, 'val_images', data_type),
                masks_path=os.path.join(self.config.preprocessed_data_path, 'val_masks', data_type),
                transform=transform
            )
        

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
            worker_init_fn=lambda _: torch.manual_seed(24)
        )

        for img, mask in data_loader:
            img, mask = self.add_single_img_processing(img=img, mask=mask)
            yield (img, mask)