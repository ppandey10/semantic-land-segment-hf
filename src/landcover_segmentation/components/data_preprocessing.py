import os
os.environ['http_proxy'] = 'http://proxy1.bgc-jena.mpg.de:3128' 
os.environ['https_proxy'] = 'http://proxy1.bgc-jena.mpg.de:3128'

import sys

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import shutil

import cv2
from PIL import Image
from patchify import patchify

import splitfolders

from landcover_segmentation.utils.common import create_directories, get_size
from landcover_segmentation.logging import logger
from landcover_segmentation.entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(
        self,
        config: DataPreprocessingConfig  
    ):
        self.config = config
        self.root_dir = config.root_dir
        self.data_path = config.data_path
        self.patch_data_path = config.patch_data_path
        self.PATCH_SIZE = config.PATCH_SIZE

        self.image_directory = os.path.join(config.data_path, 'images')
        self.mask_directory = os.path.join(config.data_path, 'masks')

    def create_patch_folder(self):
        if not os.path.exists(self.patch_data_path):
            create_directories(
                [
                    self.patch_data_path, 
                    os.path.join(self.patch_data_path, 'images'), 
                    os.path.join(self.patch_data_path, 'masks')
                ]
            )
            logger.info(f'Created {self.patch_data_path} folder along with images and masks!')

        else:
            logger.info(f'{self.patch_data_path} already exists!')

    def generate_image_patches(self):
        # folder existence check!
        if os.path.exists(self.image_directory) and os.path.exists(os.path.join(self.patch_data_path, 'images')):
            for path, subdirs, files in os.walk(self.image_directory):
                images = os.listdir(path)

                for img_name in tqdm(images, desc='Processing Images'):
                    if img_name.endswith('.tif'):
                        img = cv2.imread(os.path.join(path, img_name), 1) # read as BGR

                        SIZE_X = (img.shape[1] // self.PATCH_SIZE) * self.PATCH_SIZE
                        SIZE_Y = (img.shape[0] // self.PATCH_SIZE) * self.PATCH_SIZE

                        img = Image.fromarray(img)
                        img = img.crop((0, 0, SIZE_X, SIZE_Y))
                        img = np.array(img)

                        # create patches
                        logger.info(f'Creating patches for {img_name}')
                        img_patches = patchify(
                            image=img, 
                            patch_size=(256,256,3),
                            step=256, # no overlapping
                        )

                        for i in range(img_patches.shape[0]):
                            for j in range(img_patches.shape[1]):
                                single_img_patch = img_patches[i,j,:,:]
                                single_img_patch = single_img_patch[0]

                                # save individual patch
                                cv2.imwrite(
                                    os.path.join(self.patch_data_path, 'images', f'{os.path.splitext(img_name)[0]}_patch_{i}_{j}.tif'), single_img_patch
                                )
            logger.info('Patches stored!')

    def generate_mask_patches(self):
        # folder existence check!
        if os.path.exists(self.mask_directory) and os.path.exists(os.path.join(self.patch_data_path, 'masks')):
            for path, subdirs, files in os.walk(self.mask_directory):
                masks = os.listdir(path)

                for mask_name in tqdm(masks, desc='Processing Masks'):
                    if mask_name.endswith('.tif'):
                        mask = cv2.imread(os.path.join(path, mask_name), 0) # read as Grey

                        SIZE_X = (mask.shape[1] // self.PATCH_SIZE) * self.PATCH_SIZE
                        SIZE_Y = (mask.shape[0] // self.PATCH_SIZE) * self.PATCH_SIZE

                        mask = Image.fromarray(mask)
                        mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                        mask = np.array(mask)

                        # create patches
                        logger.info(f'Creating patches for {mask_name}')
                        mask_patches = patchify(
                            image=mask, 
                            patch_size=(256,256),
                            step=256, # no overlapping
                        )

                        for i in range(mask_patches.shape[0]):
                            for j in range(mask_patches.shape[1]):
                                single_mask_patch = mask_patches[i,j,:,:]

                                # save individual patch
                                cv2.imwrite(
                                    os.path.join(self.patch_data_path, 'masks', f'{os.path.splitext(mask_name)[0]}_patch_{i}_{j}.tif'), single_mask_patch
                                )
            logger.info('Masks patches stored!')
        else:
            sys.exit('Required folders does not exists!')
            
    def remove_unlabelled_patches(self):
        if not os.path.exists(os.path.join(self.data_path, 'labelled_dataset')):
            create_directories(
                [
                os.path.join(self.root_dir, 'labelled_dataset'),
                os.path.join(self.root_dir, 'labelled_dataset', 'images'),
                os.path.join(self.root_dir, 'labelled_dataset', 'masks')
                ]
            )


        if (
            get_size(Path(os.path.join(self.patch_data_path, 'images'))) != 0 and 
            get_size(Path(os.path.join(self.patch_data_path, 'masks'))) != 0
        ):
            useless_images = 0

            imgs_list = os.listdir(os.path.join(self.patch_data_path, 'images'))

            for img_name in tqdm(imgs_list, desc='Removing Unlabelled Images'):
                img_path = os.path.join(self.patch_data_path, 'images', img_name)
                mask_path = os.path.join(self.patch_data_path, 'masks', img_name)

                img = cv2.imread(img_path, 1)
                mask = cv2.imread(mask_path, 0)

                labels, counts = np.unique(mask, return_counts=True)

                # set storing condition
                if (1 - counts[0]/counts.sum()) > 0.05: # atleast 5% useful labels
                    cv2.imwrite(os.path.join(self.root_dir, 'labelled_dataset', 'images', img_name), img)
                    cv2.imwrite(os.path.join(self.root_dir, 'labelled_dataset', 'masks', img_name), mask)

                else:
                    useless_images += 1

            logger.info(f'Useful Images: {len(imgs_list) - useless_images}')

    def split_dataset(self):
        input_folder = os.path.join(self.root_dir, 'labelled_dataset')

        # create train and test dataset
        create_directories([os.path.join(self.root_dir, 'final_dataset')])
        self.output_folder = os.path.join(self.root_dir, 'final_dataset')

        # Split with a ratio.
        # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .1, .1)`.
        splitfolders.ratio(
            input_folder, output=self.output_folder, seed=42, ratio=(.80, .10, .10), group_prefix=None
        ) # default values

        logger.info(f'Splitting of the dataset completed!')

    def restructure_data_folder(self):
        train_images_dir = os.path.join(self.output_folder, 'train_images/train')
        train_masks_dir = os.path.join(self.output_folder, 'train_masks/train')
        val_images_dir = os.path.join(self.output_folder, 'val_images/val')
        val_masks_dir = os.path.join(self.output_folder, 'val_masks/val')
        test_images_dir = os.path.join(self.output_folder, 'test_images/test')
        test_masks_dir = os.path.join(self.output_folder, 'test_masks/test')

        create_directories(
            [
                train_images_dir,
                train_masks_dir,
                val_images_dir,
                val_masks_dir,
                test_images_dir,
                test_masks_dir
            ]
        )

        # Move the images and masks to their respective folders with progress bar
        for split in ['train', 'val', 'test']:
            for data_type in ['images', 'masks']:
                source_dir = os.path.join(self.output_folder, split, data_type)
                destination_dir = os.path.join(self.output_folder, '{}_{}'.format(split, data_type), split)

                file_list = os.listdir(source_dir)
                for file in tqdm(file_list, desc="Moving files"):
                    shutil.move(os.path.join(source_dir, file), os.path.join(destination_dir, file))

        
        # Remove the original directories created by splitfolders
        for split in ['train', 'val', 'test']:
            shutil.rmtree(os.path.join(self.output_folder, split))