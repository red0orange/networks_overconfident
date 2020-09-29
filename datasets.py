#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
'''
@author: red0orange
@file: dataset.py
@time: 2019/12/13 下午8:37
@desc:
'''

import os
import torch
from torch.utils import data
from torchvision import transforms as T
import cv2
import random


class Dataset(data.Dataset):
    def __init__(self, root, train=True, test=False, img_size=(28, 28)):
        random.seed(521)
        self.img_size = img_size
        self.data = []
        if test:
            imgs_path = os.path.join(root, 'test')
        else:
            imgs_path = os.path.join(root, 'train')
        classes = [i for i in os.listdir(imgs_path) if i.isnumeric()]
        for cls_path, cls in [(os.path.join(imgs_path, i), int(i)) for i in classes]:
            self.data.extend([(os.path.join(cls_path, i), cls) for i in os.listdir(cls_path)])
        random.shuffle(self.data)

        imgs_len = len(self.data)
        if not test and train:
            # self.data = self.data[:int(0.9*imgs_len)] #训练集
            self.data = self.data
        elif not test and not train:
            self.data = self.data[int(0.9 * imgs_len):]  # 验证集
        self.transforms = T.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        # print(img_path)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, self.img_size)
        img_tensor = torch.from_numpy(img / 255).float()
        img_tensor.unsqueeze_(0)
        return img_path, img_tensor, label
