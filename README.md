# dlcv_final_pointnet

## Data Path
將 dataset 資料夾與本專案資料夾放在同一個路徑下
```
dlcv_final_pointnet
├── dataset  # put the dataset here
├── out  # output of predictions
├── pointnet
│   ├── LICENSE
│   ├── pointnet
│   │   ├── constants.py  # scannet200 constants
│   │   ├── dataset.py  # scannet200 custom dataset
│   │   ├── model.py  # pointnet models
│   │   ├── split_train_val.py  # pre-processing
│   │   └── train.py  # training
│   ├── README.md
│   ├── scripts
│   │   ├── build.sh
│   │   └── download.sh
│   ├── setup.py
│   └── utils
│       ├── render_balls_so.cpp
│       ├── render_balls_so.so
│       ├── show3d_balls.py
│       ├── show_cls.py
│       ├── show_seg.py
│       ├── train_classification.py
│       └── train_segmentation.py
├── README.md
└── seg  # save checkpoints
```

## How to run
1. Split train/val dataset
  ```bash=
  $ cp dataset/train.txt dataset/train_origin.txt
  $ python3 pointnet/split_train_val.py
  ```
2. train
  ```bash=
  $ python3 pointnet/train.py
  ```
3. inference
  ```bash=
  $ python3 pointnet/train.py --model <checkpoint path> --nepoch -1
  ```


## Note
1. 計算 mIoU
2. 改變 model
    - dataset.py line 60 : 現在只有給 (x, y, z)
    - 嘗試 (x, y, z, r, g, b, normal values(?)) 
    - 需改變 model.py
3. 目前使用 200 labels，分別為 0 ~ 199
    - dataset.py line 67 : training 時只會給 200 labels 的資料點
    - dataset.py line 86 : validation 時會給所有資料點
    - train.py mIoU() line 108 : 計算 accuracy 時，只會計算 200 labels 的資料點
    - prediction 只會預測 200 labels，反正 200 外一定是錯的
4. train/val 切割可以從 split_train_val.py 設定 ratio
5. npoints 問題
    - train 的時候，丟統一數量的 npoints，batchsize > 1 (dataset.py line 51)
    - test/val 的時候，一次丟全部的點，但 batchsize = 1
6. dataset.py line 78,79 : number of worker = 0 
    - 因為我的 server 設number of worker = 4 會有 bug (pytorch問題)
7. TODO
    - 更加仔細的分析，各個 class 的 accuracy
    - visualization for presentation

## Possible Improvement Method
1. (x, y, z, r, g, b)
2. data augmentation (written by PointNet)
3. 200 classes -> 201 classes
4. npoint = 2500 for training -> more?
5. fine-tuning with class_choice (need more coding)
