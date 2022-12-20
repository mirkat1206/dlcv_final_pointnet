from __future__ import print_function
from torch.utils.data import Dataset
import os
import os.path
import glob
import torch
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

import constants as Kst

class ScanNet200Dataset(Dataset):
    def __init__(
            self,
            root='./dataset/', # execute this file at final/
            npoints=0,
            split='train',
            data_augmentation=True,
            class_choice=None,
        ):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.data_augmentation = data_augmentation

        self.cat = {}
        self.seg_classes = {}
        self.datapath = []

        self.num_seg_classes = 200
        
        # map scannet200_class_ids to (0 ~ 199)
        for i in range(len(Kst.VALID_CLASS_IDS_200)):
            self.cat[Kst.VALID_CLASS_IDS_200[i]] = i
        #print(self.cat)

        # special trainning on certain classes
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        # datas' filepaths
        self.datapath = [line.strip() for line in open(self.root + '/' + self.split + '.txt', 'r')]
        self.datapath = [datapath for datapath in self.datapath if len(datapath) > 0]

    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(self.root + fn, 'rb') as f:
            plydata = PlyData.read(f)
        df = pd.DataFrame(plydata.elements[0].data)
        
        ### DLCV EXPERIMENT
        # point set
        # point_set = df[['x','y','z','red','green','blue']].to_numpy()
        point_set = df[['x','y','z']].to_numpy()
        
        ### DLCV EXPERIMENT
        ###   din's method: 
        ###   > train with only 200 classes, predict only 200 classes (mIoU with only TRUE 200 classes)
        # segs : remove points with unintereseted labels (outside 200 classes)
        if self.split == 'train':
            seg = df.label.values.tolist()
            in_200 = []
            seg_200 = []
            for i in range(len(seg)):
                if seg[i] in self.cat.keys():
                    in_200.append(i)
                    seg_200.append(self.cat[seg[i]])
            point_set = point_set[in_200, :]
            seg = seg_200

            # resample
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[choice, :]
            point_set = torch.from_numpy(point_set)
            new_seg = []
            for i in choice:
                new_seg.append(seg[i])
            seg = new_seg
            seg = torch.LongTensor(seg)
        elif self.split == 'val':
            seg = df.label.values.tolist()
            seg_200 = []
            for i in range(len(seg)):
                if seg[i] in self.cat.keys():
                    seg_200.append(self.cat[seg[i]])
                else:
                    seg_200.append(-1)
            seg = seg_200
            seg = torch.LongTensor(seg)

        ### Custom Dataset & DataLoader
        # point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        # dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        # point_set = point_set / dist #scale

        # if self.data_augmentation:
        #     theta = np.random.uniform(0,np.pi*2)
        #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        #     point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
        #     point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        if self.split == 'train' or self.split == 'val':
            return point_set, seg
        else:
            print('dataset : ' , fn)
            return point_set, fn

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    dataset = ScanNet200Dataset(
        root='./dataset/', # execute this file at final/
        npoints=2500,
        class_choice=None,
        split='train',
        data_augmentation=True
    )
    point_set, seg = dataset[0]
    print(point_set)
    print(seg)