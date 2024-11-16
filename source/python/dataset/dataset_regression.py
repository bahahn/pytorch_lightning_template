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
from sklearn.model_selection import train_test_split


# %%
# Train Validate Test Split
df = pd.read_csv("./data/Student_Performance/Student_Performance.csv")

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

df_train.to_csv("./data/Student_Performance/Student_Performance_train.csv", index=False)
df_val.to_csv("./data/Student_Performance/Student_Performance_val.csv", index=False)
df_test.to_csv("./data/Student_Performance/Student_Performance_test.csv", index=False)


# %%
class CustomRegressionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        column_transformer: ColumnTransformer = None,
    ):
        self.path = dataset_path

        # Read in .csv as pandas
        self.df = pd.read_csv(self.path)

        self.X = self.df.drop(columns=["Performance Index"])
        self.y = self.df["Performance Index"]

        if column_transformer is None:
            self.prepare_column_transformer()
        else:
            self.preprocessor = column_transformer

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
        self.train_dataset = CustomRegressionDataset(
            dataset_path=dataset_path + "_train.csv"
        )
        self.val_dataset = CustomRegressionDataset(
            dataset_path=dataset_path + "_val.csv",
            column_transformer=self.train_dataset.preprocessor,
        )
        self.test_dataset = CustomRegressionDataset(
            dataset_path=dataset_path + "_test.csv",
            column_transformer=self.train_dataset.preprocessor,
        )

        self.columns = self.train_dataset.X.columns

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
# dataset = CustomRegressionDataset(
#     "./data/Student_Performance/Student_Performance_train.csv"
# )
# dataset_handler = DatasetHandler("./data/Student_Performance/Student_Performance")
# %%
# dataset[0]

# %%
