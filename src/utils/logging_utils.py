import sys
from pathlib import Path
from functools import wraps

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

def setup_logger(log_file):
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(log_file, rotation="10 MB")

def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func_name}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func_name}: {str(e)}")
            raise
    return wrapper

def get_rich_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )


#=========================================================
import torch 
import lightning as pl 
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model:pl.LightningModule,datamodule:pl.LightningDataModule):
    model.eval()
    y_pred = []
    y_true = []
    for batch in datamodule.train_dataloader():
        x,y = batch 
        logits = model(x)
        loss   = torch.nn.functional.cross_entropy(logits,y)
        preds  = torch.nn.functional.softmax(logits,dim=-1)
        # preds,true comes in batch(32)
        preds  = torch.argmax(preds,dim=-1)
        for i,j in zip(preds,y):
            print(y.shape,preds.shape,type(y),type(preds))
            y_true.append(j.item())
            y_pred.append(i.item())
    print(confusion_matrix(y_true=y_true, y_pred=y_pred))
    
# ==========================================