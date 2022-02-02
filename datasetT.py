import torch.utils.data as data
import torch
import numpy as np
import h5py
from skimage import transform,measure
# import cv2
import os
from PIL import Image
from torchvision import transforms
import random

def random_crop(moire,clean,crop_size,im_size=1024):
    if crop_size==im_size:
        return moire,clean
    else:
        rand_num_x = np.random.randint(im_size-crop_size-1)
        rand_num_y = np.random.randint(im_size-crop_size-1)
        moire = np.array(moire)
        clean = np.array(clean)
        nm = moire[rand_num_x:rand_num_x+crop_size,rand_num_y:rand_num_y+crop_size,:]
        nc = clean[rand_num_x:rand_num_x+crop_size,rand_num_y:rand_num_y+crop_size,:]
        nm = Image.fromarray(nm)
        nc = Image.fromarray(nc)
        return nm,nc

class DatasetFromImage(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromImage, self).__init__()
        ph,pl = file_path
        self.HR = os.path.join(ph)
        self.LR = os.path.join(pl)
        self.HR_list = os.listdir(self.HR)
        self.LR_list = os.listdir(self.LR)
        # print(self.HR_list[:10])
        # print(self.LR_list[:10])

    def __getitem__(self, index):
        im_h = Image.open(os.path.join(self.HR,self.HR_list[index]))
        # im_l = Image.open(os.path.join(self.LR,self.LR_list[index]))
        im_l = Image.open(os.path.join(self.LR,self.HR_list[index][:-10]+'source.png'))
        im_l,im_h = random_crop(im_l,im_h,crop_size=1024,im_size=1024)
        HR = transforms.ToTensor()(im_h)
        LR = transforms.ToTensor()(im_l)
        return LR,HR
        
    def __len__(self):
        len_h = len(self.HR_list)
        len_l = len(self.LR_list)
        if len_h >= len_l:
            len_file = len_l
        else:
            len_file = len_h
        return len_file

class MSDatasetFromImage(data.Dataset):
    def __init__(self, file_path):
        super(MSDatasetFromImage, self).__init__()
        ph1,ph2,ph4,pl = file_path
        self.HR1 = os.path.join(ph1)
        self.HR2 = os.path.join(ph2)
        self.HR4 = os.path.join(ph4)
        self.LR = os.path.join(pl)
        self.HR1_list = os.listdir(self.HR1)
        self.HR2_list = os.listdir(self.HR2)
        self.HR4_list = os.listdir(self.HR4)
        self.LR_list = os.listdir(self.LR)

    def __getitem__(self, index):
        im_h1 = Image.open(os.path.join(self.HR1,self.HR1_list[index]))
        im_h2 = Image.open(os.path.join(self.HR2,self.HR2_list[index]))
        im_h4 = Image.open(os.path.join(self.HR4,self.HR4_list[index]))
        im_l = Image.open(os.path.join(self.LR,self.LR_list[index]))
        # im_l,im_h = random_crop(im_l,im_h,crop_size=1024,im_size=1024)
        HR1 = transforms.ToTensor()(im_h1)
        HR2 = transforms.ToTensor()(im_h2)
        HR4 = transforms.ToTensor()(im_h4)
        LR = transforms.ToTensor()(im_l)
        return LR,HR1,HR2,HR4
        
    def __len__(self):
        len_h = len(self.HR1_list)
        len_l = len(self.LR_list)
        if len_h >= len_l:
            len_file = len_l
        else:
            len_file = len_h
        return len_file

def test():
    file_path = "./"
    dfi = DatasetFromImage(file_path)
