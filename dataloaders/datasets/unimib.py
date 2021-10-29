# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 13:18:51 2021

@author: udits
"""

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import glob
import os
import numpy as np
from torchvision import transforms, utils
import pandas as pd
import pdb
import cv2
import matplotlib.pyplot as plt

class UNIMIB(Dataset):
    def __init__(self, folder_path, transforms):
        super(UNIMIB, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.jpg'))
        self.mask_files = []
        self.img_names = []
        self.transforms = transforms
        #class_file = pd.read_csv('/home/us2848/UECFoodPix/UECFOODPIXCOMPLETE/data/category.txt', sep='\t')
        #self.class_ids = np.array(class_file['id'])
        #self.classs_names = np.array(class_file['name'])
        for img_path in self.img_files:
            img_name = os.path.basename(img_path)
            self.img_names.append(img_name)
            only_name = img_name[:-4]
            self.mask_files.append(os.path.join(folder_path,'masks',only_name+'.png'))
    
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        img_name = self.img_names[index]

        data = Image.open(img_path)
        label = Image.open(mask_path)
        #print(data.numpy().shape)
        #print(label.numpy().shape)
        data = self.transforms(data)
        label = self.transforms(label)
        label = label[0,:,:]*255
        sample = {'image': data, 'label': label, 'name': img_name}
        return sample

    
    def __len__(self):
        #return 1
        return len(self.img_files)
                                   
#img_size = (256,256)
#train_transform = transforms.Compose([
#		transforms.RandomHorizontalFlip(),
#   # transforms.Resize(img_size),
#        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#        transforms.ToTensor()
#        #transforms.ToPILImage()
#    ])
#val_transform = transforms.Compose([
#		#transforms.RandomHorizontalFlip(),
#    transforms.Resize(img_size),
#        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#        transforms.ToTensor()
#    ])
#                                
#train_dataset = UNIMIB(folder_path="/home/us2848/UNIMIB dataset/train/", transforms = train_transform)
#train_dataloader = DataLoader(train_dataset, batch_size=1, 
#                        shuffle=False, num_workers=1)
#
#val_dataset = UNIMIB(folder_path="/home/us2848/UNIMIB dataset/test/", transforms = val_transform)
#val_dataloader = DataLoader(val_dataset, batch_size=1,
#                        shuffle=False, num_workers=1)
#
#
#for batch in train_dataloader:
#    img = batch['image']
#    mask = batch[maskl']

