# %%
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


# %%
class CustomImageDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.path = dataset_path

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.classes = [
            name
            for name in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, name))
        ]

        dataframe_list = []
        for class_name in self.classes:
            image_file_names = [
                f
                for f in os.listdir(self.path + "/" + class_name)
                if f.lower().endswith(".jpg")
            ]

            df = pd.DataFrame(image_file_names, columns=["file_name"])

            df["class"] = class_name
            df["file_path"] = self.path + "/" + df["class"] + "/" + df["file_name"]
            dataframe_list.append(df)

        self.dataset_df = pd.concat(dataframe_list)

        # Factorize the 'class' column
        self.dataset_df["class_numeric"], self.class_mapping = pd.factorize(
            self.dataset_df["class"]
        )

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        img_path = self.dataset_df.iloc[idx]["file_path"]
        image_bw = Image.open(img_path).convert("L")
        image_tensor = self.transform(image_bw)

        class_value = self.dataset_df.iloc[idx]["class_numeric"]
        class_tensor = torch.tensor(class_value)

        return image_tensor, class_tensor

    def visualize_tensor(self, idx):
        image_tensor, class_tensor = self.__getitem__(idx)

        image = image_tensor.squeeze(0)  # Shape becomes [250, 250]

        # Plot the image
        plt.imshow(image, cmap="gray")  # Use 'gray' colormap for grayscale image
        plt.colorbar()  # Optional: shows color scale
        plt.title(f"Class: {self.class_mapping[class_tensor.item()]}")
        plt.show()

    @staticmethod
    def visualize_raw_tensor(image_tensor: torch.Tensor):
        image = image_tensor.squeeze(0)  # Shape becomes [250, 250]

        # Plot the image
        plt.imshow(image, cmap="gray")  # Use 'gray' colormap for grayscale image
        plt.colorbar()  # Optional: shows color scale
        plt.show()


class DatasetHandler:
    def __init__(self, dataset_path: str):
        self.dataset = CustomImageDataset(dataset_path=dataset_path)

        self.train_size = int(0.6 * len(self.dataset))
        self.val_size = int(0.2 * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [self.train_size, self.val_size, self.test_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)

    def get_train_val_test_split(self):
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_train_val_test_dataloaders(self):
        return self.train_loader, self.val_loader, self.test_loader


# %%
# dataset = CustomImageDataset("./data/Rice_Image_Dataset")
# dataset_handler = DatasetHandler("./data/Rice_Image_Dataset")

# # %%
# train_dataloader, _, _ = dataset_handler.get_train_val_test_dataloaders()

# tmp_batch = next(iter(train_dataloader))

# # %%
# CustomImageDataset.visualize_raw_tensor(tmp_batch[0][62])


# %%
