#!/bin/bash

while getopts g:f:p:s:n:r:z option
do 
case "${option}"
in
g) gpu=${OPTARG};;
f) config_file=${OPTARG};;
p) path_file=${OPTARG};;
s) solver=${OPTARG};;
n) n_nodes=${OPTARG};;
r) run_id=${OPTARG};;
z) zhores=1;;
esac
done

# Add packages while working on zhores
if [[ $zhores ]]
then
    home_path="/trinity/home/y.gusak"
    
    module load compilers/gcc-5.5.0 
    module add gpu/cuda-10.0
    module add python/python-3.7.1
    
    python3 -m pip install --upgrade pip --user
    python3 -m pip install numpy --user
    python3 -m pip install pandas --user
    python3 -m pip install h5py --user
    python3 -m pip install configobj --user
    
    python3 -m pip  install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl --user
    python3 -m pip  install torchvision==0.2.1 --user
    
    cd $home_path/torchdiffeq && python3 -m pip install ./ --user
    cd $home_path/interpolated_torchdiffeq && python3 -m pip install ./ --user
fi


source $config_file
echo $network $batch_size $lr $atol $rtol

source $path_file
echo $data_root $anode_dir $save_root

save="$save_root/classification_bn1-LN_resblock-LNReLU_odeblock-LNReLU/$network/${solver}_n${n_nodes}"
if [[ "$solver" == "dopri5" ]]
then
  save="${save}_tol${atol}"
fi

if [[ "$solver" == "dopri5_old" ]]
then
  save="${save}_tol${atol}"
fi

save="${save}_lr${lr}_bs${batch_size}/$run_id"
echo $save

train_file="$anode_dir/train.py"
bash -c "cd  && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python3 $train_file \
                --data_root $data_root \
                --network $network \
                --batch_size $batch_size \
                --lr $lr\
                --save $save\
                --method $solver \
                --n_nodes $n_nodes\
                --normalization_resblock 'LN'\
                --normalization_odeblock 'LN'\
                --normalization_bn1 'LN'\
                --activation_resblock 'ReLU'\
                --activation_odeblock 'ReLU'
                "

# ## Example: nohup bash scripts/train_bn1-LN_resblock-LNReLU_odeblock-LNReLU.sh -g 6 -f configs/classification_resnet10_tol1e-3_lr1e-1_bs512.cfg -p configs/pathes.py -s Euler -n 8 -r 0&
