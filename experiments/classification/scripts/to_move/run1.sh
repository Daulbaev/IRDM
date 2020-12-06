#!/bin/bash

### ResNet4, dopri5, tol: {1e-5, 1e-3, 1e-1}, (n, bs) : (8, 1024), (16, 512) 
nohup bash scripts/train.sh -g 0 -f configs/classification_resnet4_tol1e-5_lr1e-1_bs1024.cfg  -p configs/pathes.py -s dopri5 -n 8 -r 0&

nohup bash scripts/train.sh -g 1 -f configs/classification_resnet4_tol1e-3_lr1e-1_bs1024.cfg  -p configs/pathes.py -s dopri5 -n 8 -r 0&

nohup bash scripts/train.sh -g 2 -f configs/classification_resnet4_tol1e-1_lr1e-1_bs1024.cfg  -p configs/pathes.py -s dopri5 -n 8 -r 0&

nohup bash scripts/train.sh -g 3 -f configs/classification_resnet4_tol1e-5_lr1e-1_bs512.cfg  -p configs/pathes.py -s dopri5 -n 16 -r 0&

nohup bash scripts/train.sh -g 4 -f configs/classification_resnet4_tol1e-3_lr1e-1_bs512.cfg  -p configs/pathes.py -s dopri5 -n 16 -r 0&

nohup bash scripts/train.sh -g 5 -f configs/classification_resnet4_tol1e-1_lr1e-1_bs512.cfg  -p configs/pathes.py -s dopri5 -n 16 -r 0&

### ResNet4, RK2, (n, bs): (2, 1024), (4, 512) 
nohup bash scripts/train.sh -g 6 -f configs/classification_resnet4_tol1e-1_lr1e-1_bs1024.cfg  -p configs/pathes.py -s RK2 -n 2 -r 0&

nohup bash scripts/train.sh -g 7 -f configs/classification_resnet4_tol1e-1_lr1e-1_bs512.cfg  -p configs/pathes.py -s RK2 -n 4 -r 0&
