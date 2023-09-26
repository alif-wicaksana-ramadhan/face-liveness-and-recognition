import os
import cv2
import torch
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.labels_df.iloc[idx, 1]  # Assuming labels are in the second column of the CSV

        if self.transform:
            image = self.transform(image)

        return image, label


class TripletFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, device='cpu'):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        anchor_img_path = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        anchor_image = Image.open(anchor_img_path)  # Load the anchor image using OpenCV
        anchor_label = self.labels_df.iloc[idx, 1]  # Person name

        # Select a positive sample (image of the same person)
        positive_samples = self.labels_df[self.labels_df['label'] == anchor_label]
        positive_idx = random.choice(positive_samples.index.tolist())  # Convert the index to a list and then select
        positive_img_path = os.path.join(self.root_dir, self.labels_df.iloc[positive_idx, 0])
        positive_image = Image.open(positive_img_path)  # Load the positive image using OpenCV

        # Select a negative sample (image of a different person)
        negative_samples = self.labels_df[self.labels_df['label'] != anchor_label]
        negative_idx = random.choice(negative_samples.index.tolist())  # Convert the index to a list and then select
        negative_img_path = os.path.join(self.root_dir, self.labels_df.iloc[negative_idx, 0])
        negative_image = Image.open(negative_img_path)  # Load the negative image using OpenCV

        # Convert images to RGB format, PyTorch tensor, and normalize
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image