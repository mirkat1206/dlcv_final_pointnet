from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ScanNet200Dataset

import constants as Kst
import sys
sys.path.append('.')

from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='./out/', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--class_choice', type=str, default=None, help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)


# Utils
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device used: ", device)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


# Custom Dataset & DataLoader
train_npoints = 2500 ### DLCV EXPERIMENT
trainset = ScanNet200Dataset(
    root='./dataset/', # execute this file at final/
    npoints=train_npoints,
    class_choice=opt.class_choice,
    split='train',
    data_augmentation=True
)

valset = ScanNet200Dataset(
    root='./dataset/', # execute this file at final/
    npoints=0,
    class_choice=None,
    split='val',
    data_augmentation=False
)

testset = ScanNet200Dataset(
    root='./dataset/', # execute this file at final/
    npoints=0,
    class_choice=None,
    split='test',
    data_augmentation=False
)

dataloader = {
    'train': DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers)),
    'val': DataLoader(valset, batch_size=1, shuffle=True, num_workers=int(0), drop_last=True),
    'test': DataLoader(testset, batch_size=1, shuffle=False, num_workers=int(0)),
}

print(len(trainset), len(valset), len(testset))


### DLCV EXPERIMENT
# num_classes = trainset.num_seg_classes + 1
num_classes = trainset.num_seg_classes 
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass


# Model
classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
# classifier = nn.DataParallel(classifier)
classifier.to(device)
if opt.model != '':
    print('load model from ', opt.model)
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

num_batch = len(trainset) / opt.batchSize

def mIoU(pred_np, target_np):
    ### DLCV TODO : mIoU calculation?
    total_correct = 0
    total_points = 0
    for i in range(len(target_np)):
        if target_np[i] != -1:
            total_points += 1
            if target_np[i] == pred_np[i]:
                total_correct += 1
    return total_correct, total_points
            

def val(max_i=-1):
    # set max_i to save validation time
    # max_i = how many validation .ply files are evaluated
    classifier.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader['val'], 0):
            if i > max_i and max_i != -1:
                break
            points, target = data
            # # [batch_size, num_points, channels] - >[batch_size, channels, num_points]
            points = points.transpose(2, 1) 
            points, target = points.to(device), target.to(device)
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            pred_choice = pred.data.max(1)[1]
            
            pred_np = pred_choice.cpu().data.numpy()
            target_np = target.cpu().data.numpy()
            total_correct, total_points = mIoU(pred_np, target_np)            
        print('\n\tvalidation accuracy: %f\n' % (
            total_correct / float(total_points)
        ))



def test():
    classifier.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader['test'], 0):
            points, fn = data
            fn = os.path.basename(fn[0])
            fn = fn[:fn.find('.')]
            # [batch_size, num_points, channels] - >[batch_size, channels, num_points]
            points = points.transpose(2, 1) 
            points = points.to(device)
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            pred_choice = pred.data.max(1)[1]
            
            pred_np = pred_choice.cpu().data.numpy()
            with open(opt.outf + fn + '.txt', 'w') as f:
                for j in range(len(pred_np)):
                    f.write(str(Kst.VALID_CLASS_IDS_200[pred_np[j]]))
                    f.write(' ')
                    f.write(str(Kst.CLASS_LABELS_200[pred_np[j]]))
                    f.write('\n')


for epoch in range(opt.nepoch):
    print('------')
    for i, data in enumerate(dataloader['train'], 0):
        points, target = data
        # [batch_size, num_points, channels] - >[batch_size, channels, num_points]
        points = points.transpose(2, 1) 
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, 
            i, 
            num_batch, 
            loss.item(), 
            correct.item() / float(points.shape[0] * train_npoints)
        ))

        if i % 10 == 9:
            val(max_i=100)
            
    scheduler.step()
    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

val()
test()

# ## benchmark mIOU
# shape_ious = []
# for i, data in enumerate(dataloader['train']):
#     points, target = data
#     points = points.transpose(2, 1)
#     points, target = points.cuda(), target.cuda()
#     classifier = classifier.eval()
#     pred, _, _ = classifier(points)
#     pred_choice = pred.data.max(2)[1]

#     pred_np = pred_choice.cpu().data.numpy()
#     target_np = target.cpu().data.numpy()

#     for shape_idx in range(target_np.shape[0]):
#         parts = range(num_classes)#np.unique(target_np[shape_idx])
#         part_ious = []
#         for part in parts:
#             I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             if U == 0:
#                 iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / float(U)
#             part_ious.append(iou)
#         shape_ious.append(np.mean(part_ious))

# print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))