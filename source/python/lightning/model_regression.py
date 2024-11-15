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
)


from source.python.dataset.dataset_regression import DatasetHandler


# %%
class LinearNN(LightningModule):

    def __init__(self, num_features_in: int):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(num_features_in, 256)
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
            self.parameters(), lr=0.0005, weight_decay=0.001
        )
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "validation_mse_epoch",
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


# %%
dataset_handler = DatasetHandler("./data/Student_Performance.csv")
dataset_handler.columns

# %%
model = LinearNN(num_features_in=6)
# %%
# Logger
logger = TensorBoardLogger("lightning_logs", "LNN")
metrics = {}
for metric in list(model.base_metrics.keys()):
    metrics[metric] = [
        "Multiline",
        [f"train_{metric}_epoch", f"validation_{metric}_epoch"],
    ]

tensorboard_layout = {"Statistics": metrics}
logger.experiment.add_custom_scalars(tensorboard_layout)

csv_logger = CSVLogger("csv_logs", "LNN")

# %%
early_stopping = EarlyStopping(monitor="validation_mse_epoch", patience=6)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd() + "/models/LNN/",
    save_top_k=1,
    verbose=True,
    monitor="validation_mse_epoch",
    mode="min",
)

lr_logger = LearningRateMonitor()


# %%
trainer = pl.Trainer(
    max_epochs=50,
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
        ModelSummary(max_depth=1),
    ],
    accelerator="mps",
)
trainer.fit(model, dataset_handler.train_loader, dataset_handler.val_loader)

# %%

# %%
trainer.test(model, dataset_handler.test_loader)

# %%
x_1, y_1 = dataset_handler.test_dataset[23]

y_pred = model.forward(x_1)
print("True: ", y_1)
print(
    "Predicted: ",
    y_pred,
)

# %%
