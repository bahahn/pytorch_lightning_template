# %%
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# %%
class CustomRegressionDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.path = dataset_path

        # Read in .csv as pandas
        self.df = pd.read_csv("./data/Student_Performance.csv")

        self.X = self.df.drop(columns=["Performance Index"])
        self.y = self.df["Performance Index"]

        self.prepare_column_transformer()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data_part = self.X.iloc[[idx]]
        target_part = self.y.iloc[idx]

        # Use column transformer
        data_part_ct = self.preprocessor.transform(data_part)

        return (
            torch.from_numpy(data_part_ct).float().view(-1),
            torch.tensor(target_part).float(),
        )

    def prepare_column_transformer(self):
        numeric_column_names = self.X.select_dtypes(include="number").columns
        categorical_column_names = self.X.select_dtypes(
            include=["object", "category"]
        ).columns

        # Create pipelines for numeric and categorical features
        numeric_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standard scaling for numeric features
                #         ("onehot", OneHotEncoder()),  # One-hot encoding for scaled numeric features
            ]
        )

        categorical_pipeline = Pipeline(
            [("onehot", OneHotEncoder())]  # One-hot encoding for categorical features
        )

        # Combine pipelines in a column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    numeric_column_names,
                ),  # Apply numeric pipeline
                (
                    "cat",
                    categorical_pipeline,
                    categorical_column_names,
                ),  # Apply categorical pipeline
            ]
        )

        self.preprocessor = self.preprocessor.fit(self.X)


class DatasetHandler:
    def __init__(self, dataset_path: str, batch_size: int = 128):
        self.dataset = CustomRegressionDataset(dataset_path=dataset_path)
        self.columns = self.dataset.X.columns

        self.train_size = int(0.6 * len(self.dataset))
        self.val_size = int(0.2 * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [self.train_size, self.val_size, self.test_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

    def get_train_val_test_split(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_train_val_test_dataloaders(self):
        return self.train_loader, self.val_loader, self.test_loader


# %%
dataset = CustomRegressionDataset("./data/Student_Performance.csv")
# dataset_handler = DatasetHandler("./data/Student_Performance.csv")
# %%
dataset[0]

# %%
