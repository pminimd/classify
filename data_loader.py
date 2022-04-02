import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import cv2

class Mura_Classify_Dataset(Dataset):
    """Mura Classify dataset."""

    def __init__(self, mura_dataset_dir, transform=None):
        """
        Args:
            csv_file (string): img_paths_with_annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs_dir = os.path.join(mura_dataset_dir,"mura")
        self.annotations_file_path = os.path.join(mura_dataset_dir,"filename.txt")
        self.img_paths_with_annotations = pd.read_csv(self.annotations_file_path)
        self.transform = transforms.Compose([transforms.ToTensor()]) # range [0, 255] -> [0.0,1.0] and convert [H,W,C] to [C,H,W]

    def __len__(self):
        return len(self.img_paths_with_annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgpath_with_annotation = os.path.join(self.imgs_dir,
                                self.img_paths_with_annotations.iloc[idx, 0])
        img_path, label = imgpath_with_annotation.split()
        # print(img_path)
        # print(label)
        image = cv2.imread(img_path,-1)
        # print(image)
        min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(image)
        image = (image-min_val)*255/(max_val-min_val+1)
        # print(image)
        image = np.uint8(image)
        # print(image)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        
        # if label == "1":
        #     label = torch.tensor([[1,0]])
        # else:
        #     label = torch.tensor([[0,1]])
        # print(image)
        # image = np.float32(image)
        # image = plt.imread(img_path)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)

        if self.transform:
            image = self.transform(image)

        return image, int(label)

if __name__ == '__main__':
    dataset = Mura_Classify_Dataset("/home/wenjun/Documents/检测网络/Mura_Dataset")
    print(dataset[1][0],dataset[1][1])
    # for i in range(len(dataset)):
    #     item = dataset[i]
    #     print(item[0], item[1])