import os
os.environ['http_proxy'] = 'http://proxy1.bgc-jena.mpg.de:3128' 
os.environ['https_proxy'] = 'http://proxy1.bgc-jena.mpg.de:3128'

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import segmentation_models_pytorch as smp

from landcover_segmentation.logging import logger
from landcover_segmentation.entity import ModelTrainerConfig
from landcover_segmentation.pipeline.step_03_data_loader import DataLoaderPipeline

class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig
    ):
        self.config = config

        self.data_loader_pipeline = DataLoaderPipeline()
        self.generators = self.data_loader_pipeline.main()

    def smp_model(self): # TODO: include other models as well
        return smp.Unet(
            encoder_name=self.config.BACKBONE,
            encoder_weights=self.config.encoder_weights,
            in_channels=3, classes=self.config.n_classes, 
            activation=self.config.activation
        )
    
    def loss_fn(self, out, mask):
        if self.config.loss == 'jaccard':
            loss_fnc = smp.losses.JaccardLoss(mode=smp.losses.MULTILABEL_MODE)
        
        if self.config.loss == 'dice':
            loss_fnc = smp.losses.DiceLoss(mode=smp.losses.MULTILABEL_MODE)
        
        return loss_fnc(out, mask)
    
    def metrics(self, out, targ):

        METRICS = ['iou', 'accuracy', 'f1', 'f2', 'precision']

        tp, fp, fn, tn = smp.metrics.get_stats(out, targ, mode='multilabel', threshold=0.5)

        if self.config.metrics == 'iou':
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        return iou_score
    
    def optimizer(self):
        if self.config.optimizer == 'adam':
            opt = optim.Adam(self.smp_model().parameters(), lr=0.001)

        return opt
    
    def train_loader(self):
        train_data_loader = self.generators['train']
        try:
            next(train_data_loader)
        except StopIteration:
            pass

        return train_data_loader, self.data_loader_pipeline.data_loader._total_batches
    
    def val_loader(self):
        val_data_loader = self.generators['val']
        try:
            next(val_data_loader)
        except StopIteration:
            pass

        return val_data_loader, self.data_loader_pipeline.data_loader._total_batches
    
    def test_loader(self):
        test_data_loader = self.generators['test']
        try:
            next(test_data_loader)
        except StopIteration:
            pass

        return test_data_loader, self.data_loader_pipeline.data_loader._total_batches
    
    def model_trainer(
        self,
        device: str = None
    ):
        if device=='cpu':
            self.config.DEVICE = device
        
        self.model = self.smp_model()
        self.model.double().to(device)

        train_dataloader, train_total_batch = self.train_loader()
        val_dataloader, val_total_batch = self.val_loader()
        test_dataloader, test_total_batch = self.test_loader()

        # train model
        num_epochs = self.config.epochs
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            train_iou = 0
            with tqdm(total=test_total_batch, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for images, masks in test_dataloader:
                    images = images.double().to(device)
                    masks = masks.double().to(device)
                    
                    self.optimizer().zero_grad()
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
                    loss.backward()
                    self.optimizer().step()
                    epoch_loss += loss.item()

                    train_iou_val = self.metrics(outputs, masks)
                    train_iou += train_iou_val
                    
                    pbar.set_postfix({"loss": epoch_loss, "iou": train_iou_val})
                    pbar.update(1)

            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/test_total_batch}")


            # Validation
            self.model.eval()
            val_loss = 0
            epoch_iou_val = 0
            with tqdm(total=val_total_batch, desc=f"Validation {epoch+1}/{num_epochs}", unit="batch") as pbar:
                with torch.no_grad():
                    for images, masks in val_dataloader:
                        images = images.double().to(device)
                        masks = masks.double().to(device)
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, masks)
                        val_loss += loss.item()
                        
                        pbar.set_postfix({"val_loss": val_loss/val_total_batch})
                        pbar.update(1)

            logger.info(f"Validation Loss: {val_loss/val_total_batch},  IoU: {epoch_iou_val/val_total_batch}")

    def save_model(self):
        self.model.save_pretrained(self.config.root_dir)
