from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
# Ignore warnings
import warnings
# warnings.filterwarnings("ignore")
#
# plt.ion()   # interactive mode
#
# digit_target_dataframe = pd.read_csv('digit_module/data/train/digit_target.csv')
#



# def show_target(n):
#     """Show image with landmarks"""
#     image_name = digit_target_dataframe.iloc[n, 0]
#     target = digit_target_dataframe.iloc[n, 1]
#     image_path = io.imread(os.path.join('data/train/', image_name))
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.set_title(image_name+"---target:" + target)
#     plt.imshow(image_path)
#
#     plt.pause(0.001)  # pause a bit so that plots are updated
#

# show_target(1)

class DigitDataset(Dataset):
    def __init__(self,root_dir_of_dataset,train,transform=transforms.Compose(
    [transforms.ToTensor()])):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir_of_dataset = root_dir_of_dataset
        self.transform = transform
        self.train = train
        if self.train == "True":
            csv_path = os.path.join(self.root_dir_of_dataset,
                                "train/digit_target.csv")
            self.digit_target_dataframe = pd.read_csv(csv_path)
        else:
            csv_path = os.path.join(self.root_dir_of_dataset,
                                "test/digit_target.csv")
            self.digit_target_dataframe = pd.read_csv(csv_path)


    def __len__(self):
        return len(self.digit_target_dataframe)

    def __getitem__(self, idx):
        if self.train == True:
            image_path =  os.path.join(self.root_dir_of_dataset+"/train/" ,
                                    self.digit_target_dataframe.iloc[idx, 0])
        else:
            image_path =  os.path.join(self.root_dir_of_dataset+"/test/" ,
                                    self.digit_target_dataframe.iloc[idx, 0])
        print(image_path)
        image = np.array(io.imread(image_path))
        image = Image.fromarray(image).convert('L')
        target = int(self.digit_target_dataframe.iloc[idx, 1])
        if self.transform is not None:
            image = self.transform(image)
        return image,target


# fig = plt.figure()
#
# for i in range(len(digit_dataset)):
#     sample = digit_dataset[i]
#
#     print(i, sample['image'].shape, sample['target'])
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_target(i)
#
#     if i == 3:
#         plt.show()
#         break

#
# for i in range(len(digit_dataset)):
#     image,target = digit_dataset[i]
#     print(i, image.size(), target)
#     if i == 3:
#         break