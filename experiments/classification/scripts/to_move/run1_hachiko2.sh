#!/bin/bash

### ResNet4, {dopri5, dopri5_old}, tol: {1e-5}, bs = 64
nohup bash scripts/train.sh -g 2 -f configs/classification_resnet4_tol1e-5_lr1e-1_bs64.cfg  -p configs/pathes.py -s dopri5_old -n 0 -r 0&

nohup bash scripts/train.sh -g 3 -f configs/classification_resnet4_tol1e-5_lr1e-1_bs64.cfg  -p configs/pathes.py -s dopri5 -n 16 -r 0&

### ResNet4, Euler: (n = 4, bs = 1024), (n = 8, bs = 512)

nohup bash scripts/train.sh -g 4 -f configs/classification_resnet4_tol1e-5_lr1e-1_bs1024.cfg  -p configs/pathes.py -s Euler -n 4 -r 0&

nohup bash scripts/train.sh -g 5 -f configs/classification_resnet4_tol1e-5_lr1e-1_bs512.cfg  -p configs/pathes.py -s Euler -n 8 -r 0&


### ResNet10, RK2: (n = 2, bs = 512), Euler: (n = 8, bs = 512)

nohup bash scripts/train.sh -g 6 -f configs/classification_resnet10_tol1e-5_lr1e-1_bs512.cfg  -p configs/pathes.py -s RK2 -n 4 -r 0&

nohup bash scripts/train.sh -g 7 -f configs/classification_resnet10_tol1e-5_lr1e-1_bs512.cfg  -p configs/pathes.py -s Euler -n 8 -r 0&

