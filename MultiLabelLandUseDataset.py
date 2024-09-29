import torch
from torch.utils.data import DataLoader,Dataset,random_split
from PIL import Image
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F

class MultiLabelLandUseDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(label_file, delimiter='\t')
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get image name from the labels dataframe
        img_name = self.labels_df.iloc[idx, 0] + '.tif'  # Add file extension
        img_path = None

        # Loop through folders and subfolders to find the image
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file == img_name:
                    img_path = os.path.join(root, file)
                    break  # Stop searching once the image is found

        # Check if img_path was set (i.e., the image was found)
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in any of the folders")

        # Open the image
        image = Image.open(img_path).convert('RGB')

        # Get the corresponding label (multi-label vector)
        label = self.labels_df.iloc[idx, 1:].values.astype('float')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

