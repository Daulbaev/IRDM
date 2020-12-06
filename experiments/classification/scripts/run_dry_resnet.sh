#!/bin/bash
network="resnet34"
inplanes=64
batch_size=512


anode_dir="/trinity/home/y.gusak/anode_cheb/"
run_file="${anode_dir}scripts/run_dry.sh"

training_config="${anode_dir}configs/training_lr1e-1_bs${batch_size}.cfg"
solver_config="${anode_dir}configs/solver_tol1e-3.cfg"
model_config="${anode_dir}configs/model_${network}_inplanes${inplanes}.cfg"
normact_config="${anode_dir}configs/normact_bn1-BN_resblock-BNReLU_odeblock-LNReLU.cfg"
path_file="${anode_dir}configs/pathes_zhores_dry.py"


#echo $training_config $solver_config $model_config $normact_config $path_file

### {ResNet6}, gpu = 0, {Euler, RK2}

nohup cd $anode_dir &&\
bash $run_file -g 0 -t $training_config -c $solver_config -m $model_config -a $normact_config -p $path_file -s Euler -r 0 || \
bash $run_file -g 0 -t $training_config -c $solver_config -m $model_config -a $normact_config -p $path_file -s RK2 -r 0&

### ResNet6, gpu=1, dopri5
nohup cd $anode_dir &&\ 
bash $run_file -g 1 -t $training_config -c $solver_config -m $model_config -a $normact_config -p $path_file -s dopri5_old_cheb -r 0&


### ResNet6, gpu2, RK4, dopri5_old
nohup cd $anode_dir &&\
bash $run_file -g 2 -t $training_config -c $solver_config -m $model_config -a $normact_config -p $path_file -s RK4 -r 0 || \
bash $run_file -g 2 -t $training_config -c $solver_config -m $model_config -a $normact_config -p $path_file -s dopri5_old -r 0&


### ResNet6, gpu3, dopri5
nohup cd $anode_dir &&\
bash $run_file -g 3 -t $training_config -c $solver_config -m $model_config -a $normact_config -p $path_file -s dopri5 -r 0&


