#!/bin/bash

while getopts g:t:c:m:a:o:i:p:s:n:r:z option
do 
case "${option}"
in
g) gpu=${OPTARG};;
t) training_config=${OPTARG};;
c) solver_config=${OPTARG};;
m) model_config=${OPTARG};;
a) act_config=${OPTARG};;
o) norm_config=${OPTARG};;
i) paramnorm_config=${OPTARG};;
p) path_file=${OPTARG};;
s) solver=${OPTARG};;
n) n_nodes=${OPTARG};;
r) torch_seed=${OPTARG};;
z) zhores=1;;
esac
done

# Add packages while working on zhores
if [[ $zhores ]]
then
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

    python3 -m pip install git+https://github.com/rtqichen/torchdiffeq.git --user
    python3 -m pip install git+https://github.com/Daulbaev/interpolated_torchdiffeq.git --user
fi

source $training_config
echo $batch_size $lr

source $solver_config
echo $atol $rtol

source $model_config
echo $network $inplanes

source $act_config
echo   $activation_resblock $activation_odeblock $activation_bn1

source $norm_config
echo  $normalization_resblock $normalization_odeblock $normalization_bn1 

source $paramnorm_config
echo  $param_normalization_resblock $param_normalization_odeblock $param_normalization_bn1

source $path_file
echo $data_root $anode_dir $save_root

num_epochs=350

save="$save_root/classification_bn1-${param_normalization_bn1}${normalization_bn1}${activation_bn1}_resblock-${param_normalization_resblock}${normalization_resblock}${activation_resblock}_odeblock-${param_normalization_odeblock}${normalization_odeblock}${activation_odeblock}/${network}_inplanes${inplanes}"

train_file="$anode_dir/train.py"

command_full="cd $anode_dir "


if [[ $zhores ]]
then
    command_cuda=""
else
    #command_cuda="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu"
    command_cuda=""
fi

        echo $n
	
        if [[ "$solver" == "dopri5" ]] || [[ "$solver" == "dopri5_old" ]]
        then
          save_curr="${save}/${solver}_n${n_nodes}_tol${atol}${rtol}"
        else
          save_curr="${save}/${solver}_n${n_nodes}"
        fi
	
        save_curr="${save_curr}_lr${lr}_bs${batch_size}/$torch_seed"
        echo $save_curr

        command=" && $command_cuda python3 $train_file \
                 --data_root $data_root \
                 --network $network \
                 --batch_size $batch_size \
                 --lr $lr\
                 --save $save_curr\
                 --method $solver \
                 --n_nodes $n_nodes \
                 --inplanes $inplanes \
                 --normalization_resblock $normalization_resblock\
                 --normalization_odeblock $normalization_odeblock\
                 --normalization_bn1 $normalization_bn1\
                 --param_normalization_resblock $param_normalization_resblock\
                 --param_normalization_odeblock $param_normalization_odeblock\
                 --param_normalization_bn1 $param_normalization_bn1\
                 --activation_resblock $activation_resblock\
                 --activation_odeblock $activation_odeblock\
                 --activation_bn1 $activation_bn1\
                 --atol $atol\
                 --rtol $rtol\
                 --atol_scheduler '{150: 1.0, 250: 1.0}'\
                 --rtol_scheduler '{150: 1.0, 250: 1.0}'\
                 --num_epochs $num_epochs\
                 --torch_seed $torch_seed"
	command_full="$command_full $command"


# command_full="$command_full;"
# echo $command_full

bash -c "$command_full"

# ## Example: nohup bash scripts/run_workshop.sh   -t configs/training_lr1e-1_bs512.cfg -m configs/model_resnet10_inplanes64.cfg -a configs/act_bn1-ReLU_resblock-ReLU_odeblock-ReLU.cfg -o configs/norm_bn1-NormFree_resblock-NormFree_odeblock-NormFree.cfg -i configs/paramnorm_bn1-ParamNormFree_resblock-ParamNormFree_odeblock-ParamNormFree.cfg -p configs/pathes_hachiko_workshop.py -c configs/solver_tol1e-3.cfg -s Euler -r 502 -g 0 -n 8&

### Example(run from an node) sbatch -p gpu_devel --gpus=1 -D ~/classification/slurm_logs ~/classification/scripts/run_workshop.sh  -t ~/classification/configs/training_lr1e-1_bs512.cfg -m ~/classification/configs/model_resnet10_inplanes64.cfg -a ~/classification/configs/act_bn1-ReLU_resblock-ReLU_odeblock-ReLU.cfg -o ~/classification/configs/norm_bn1-NormFree_resblock-NormFree_odeblock-NormFree.cfg -i ~/classification/configs/paramnorm_bn1-ParamNormFree_resblock-ParamNormFree_odeblock-ParamNormFree.cfg -p ~/classification/configs/pathes_zhores_workshop.py -c configs/solver_tol1e-3.cfg -s Euler -r 502 -g 0 -n 8
