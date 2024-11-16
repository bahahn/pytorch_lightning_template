# %%
import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from lightning import LightningModule
from lightning import Trainer
import lightning as pl
import matplotlib.pyplot as plt
from PIL import Image
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MetricCollection,
    MeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    R2Score,
)
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)


from source.python.dataset.dataset_regression import DatasetHandler
from source.python.configs.config_reader import read_yml_to_dict


# %%
class LinearNN(LightningModule):

    def __init__(self, hparams_dict: dict = {}):
        super(LinearNN, self).__init__()

        # Hyperparameters
        self.hparams_dict = hparams_dict
        self.save_hyperparameters(hparams_dict, logger=True)

        self.fc1 = nn.Linear(self.hparams_dict["num_features_in"], 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, 512)
        self.fc6 = nn.Linear(512, 50)
        self.fc7 = nn.Linear(50, 1)

        # Classification Scores
        self.base_metrics = MetricCollection(
            {
                "mse": MeanSquaredError(),
                "r2": R2Score(),
                "mape": MeanAbsolutePercentageError(),
                "mae": MeanAbsoluteError(),
            },
        )
        self.train_metrics = self.base_metrics.clone(prefix="train_")
        self.validation_metrics = self.base_metrics.clone(prefix="validation_")
        self.test_metrics = self.base_metrics.clone(prefix="test_")

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.relu(self.fc4(X))
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = self.fc7(X)
        return X

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams_dict["optimizer"]["lr"],
            weight_decay=self.hparams_dict["optimizer"]["weight_decay"],
        )
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.hparams_dict["scheduler"]["factor"],
                patience=self.hparams_dict["scheduler"]["patience"],
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": self.hparams_dict["scheduler"]["monitor"],
        }

        return [self.optimizer], [self.scheduler]

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_pred = self(X).view(-1)

        metrics = self.train_metrics(y_pred, y)
        # metrics["train_cross_entropy_loss"] = F.cross_entropy(y_pred, y)
        self.log_dict(metrics, on_epoch=True, on_step=True, prog_bar=True)

        return metrics["train_mse"]

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_pred = self(X).view(-1)
        metrics = self.validation_metrics(y_pred, y)
        # metrics["validation_cross_entropy_loss"] = F.cross_entropy(y_pred, y)
        self.log_dict(metrics, on_epoch=True, on_step=True, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_pred = self(X).view(-1)
        metrics = self.test_metrics(y_pred, y)
        # metrics["test_cross_entropy_loss"] = F.cross_entropy(y_pred, y)
        self.log_dict(metrics, on_epoch=True, on_step=True, prog_bar=True)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass


# %%
config_dict = read_yml_to_dict("source/python/configs/params.yaml")
config_dict["num_features_in"] = 6

# %%
dataset_handler = DatasetHandler(
    "./data/Student_Performance.csv", batch_size=config_dict["batch_size"]
)
dataset_handler.columns

# %%
model = LinearNN(hparams_dict=config_dict)

# %%
# Logger
logger = TensorBoardLogger("logs/lightning_logs", config_dict["model_name"])
metrics = {}
for metric in list(model.base_metrics.keys()):
    metrics[metric] = [
        "Multiline",
        [f"train_{metric}_epoch", f"validation_{metric}_epoch"],
    ]

tensorboard_layout = {"Statistics": metrics}
logger.experiment.add_custom_scalars(tensorboard_layout)

csv_logger = CSVLogger("logs/csv_logs", config_dict["model_name"])

# %%
early_stopping = EarlyStopping(
    monitor=config_dict["early_stopping"]["monitor"],
    patience=config_dict["early_stopping"]["patience"],
)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd() + f"/models/{config_dict["model_name"]}/",
    save_top_k=1,
    verbose=True,
    monitor="validation_mse_epoch",
    mode="min",
)

lr_logger = LearningRateMonitor()

tqdm_bar = TQDMProgressBar(leave=True)


# %%
trainer = pl.Trainer(
    max_epochs=config_dict["num_max_epochs"],
    enable_progress_bar=True,
    benchmark=True,
    fast_dev_run=False,
    log_every_n_steps=1,
    logger=[logger, csv_logger],
    profiler="pytorch",
    callbacks=[
        early_stopping,
        checkpoint_callback,
        lr_logger,
        tqdm_bar,
        ModelSummary(max_depth=1),
    ],
    accelerator="mps",
)
trainer.fit(model, dataset_handler.train_loader, dataset_handler.val_loader)

# %%
model.eval()
trainer.test(model, dataset_handler.test_loader)


# %%
# Save model
torch.save(
    model.state_dict(),
    f"./models/{config_dict["model_name"]}/{config_dict["model_name"]}_final.pt",
)

# Save model structure
torch.save(
    model,
    f"./models/{config_dict["model_name"]}/{config_dict["model_name"]}_final_full.pt",
)


# %%
model = LinearNN(hparams_dict=config_dict)
model.load_state_dict(
    torch.load(
        f"./models/{config_dict["model_name"]}/{config_dict["model_name"]}_final.pt"
    )
)
model.eval()
# %%

# %%

# %%
x_1, y_1 = dataset_handler.test_dataset[10]

y_pred = model(x_1)
print("True: ", y_1)
print(
    "Predicted: ",
    y_pred,
)

# %%
