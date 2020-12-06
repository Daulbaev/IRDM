#!/bin/bash

### ResNet10, dopri5, tol = 1e-5, n: {8, 16, 24}, bs = 512
nohup bash scripts/train.sh -g 0 -f configs/classification_resnet10_tol1e-5_lr1e-1_bs512.cfg  -p configs/pathes.py -s dopri5 -n 8 -r 0&

nohup bash scripts/train.sh -g 1 -f configs/classification_resnet10_tol1e-5_lr1e-1_bs512.cfg  -p configs/pathes.py -s dopri5 -n 16 -r 0&

nohup bash scripts/train.sh -g 2 -f configs/classification_resnet10_tol1e-5_lr1e-1_bs512.cfg  -p configs/pathes.py -s dopri5 -n 20 -r 0&

### ResNet10, dopri5_old, tol = 1e-5, bs = 512
nohup bash scripts/train.sh -g 3 -f configs/classification_resnet10_tol1e-5_lr1e-1_bs512.cfg  -p configs/pathes.py -s dopri5_old -n 0 -r 0&


