import lightning as pl 
import timm 
import torch
from torch import nn 
from torch.nn import functional as F
from torchmetrics import Accuracy
from typing import Dict

class DogsBreedClassifier(pl.LightningModule):
    def __init__(
            self, 
            model_name:str, 
            num_classes:int, 
            pretrained:bool,
            trainable:bool,
            lr:float, 
            weight_decay:float,
            scheduler_factor:float,
            scheduler_patience:int,
            min_lr:float
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model:timm.models.resnest.ResNet = timm.create_model(model_name=self.hparams.model_name,pretrained=self.hparams.pretrained,num_classes=self.hparams.num_classes,global_pool = 'avg')

        # for p in self.model.parameters():
        #     p.requires_grad=self.hparams.trainable


        self.train_acc:Accuracy = Accuracy(task='multiclass',num_classes=self.hparams.num_classes)
        self.test_acc :Accuracy = Accuracy(task='multiclass',num_classes=self.hparams.num_classes)
        self.valid_acc:Accuracy = Accuracy(task='multiclass',num_classes=self.hparams.num_classes)


    def forward(self, x:torch.Tensor) -> torch.Any:
        return self.model(x)
    
    def training_step(self, batch,batch_idx) -> torch.Tensor:
        x,y = batch 
        logits = self(x)
        loss   = F.cross_entropy(logits,y)
        preds  = F.softmax(logits,dim=-1)
        self.train_acc(preds,y)
        self.log("train/loss",loss,prog_bar=True,on_epoch=True,on_step=True)
        self.log("train/acc",self.train_acc,prog_bar=True,on_epoch=True,on_step=True)
        return loss 

    
    def validation_step(self, batch,batch_idx) -> torch.Tensor:
        x,y = batch 
        logits = self(x)
        loss   = F.cross_entropy(logits,y)
        preds  = F.softmax(logits,dim=-1)
        self.valid_acc(preds,y)
        self.log("val/loss",loss,prog_bar=True,on_epoch=True,on_step=True)
        self.log("val/acc",self.valid_acc,prog_bar=True,on_epoch=True,on_step=True)
        return loss 
    

    def test_step(self,batch,batch_idx ) -> torch.Tensor:
        x,y = batch 
        logits = self(x)
        loss   = F.cross_entropy(logits,y)
        preds  = F.softmax(logits,dim=-1)
        self.test_acc(preds,y)
        self.log("test/loss",loss,prog_bar=True,on_epoch=True,on_step=True)
        self.log("test/acc",self.test_acc,prog_bar=True,on_epoch=True,on_step=True)
        return loss 

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(
                                    self.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay
                    )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #                 optimizer=optimizer,
        #                 factor=self.hparams.scheduler_factor,
        #                 patience=self.hparams.scheduler_patience,
        #                 min_lr=self.hparams.min_lr
        # )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,total_steps=self.trainer.estimated_stepping_batches,max_lr=self.hparams.lr*10)
        return {
            "optimizer":optimizer,
            "lr_scheduler":scheduler,
            "monitor":"train/loss_epoch"
        }
    