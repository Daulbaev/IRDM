#!/bin/bash

while getopts g:t:c:m:a:p:s:r:z option
do 
case "${option}"
in
g) gpu=${OPTARG};;
t) training_config=${OPTARG};;
c) solver_config=${OPTARG};;
m) model_config=${OPTARG};;
a) normact_config=${OPTARG};;
p) path_file=${OPTARG};;
s) solver=${OPTARG};;
r) run_id=${OPTARG};;
z) zhores=1;;
esac
done

source $training_config
echo $batch_size $lr

source $solver_config
echo $atol $rtol

source $model_config
echo $network $inplanes

source $normact_config
echo  $normalization_resblock $normalization_odeblock $normalization_bn1 $activation_resblock $activation_odeblock

source $path_file
echo $data_root $anode_dir $save_root


save="$save_root/classification_bn1-${normalization_bn1}_resblock-${normalization_resblock}${activation_resblock}_odeblock-${normalization_odeblock}${activation_odeblock}/${network}_inplanes${inplanes}"

train_file="$anode_dir/train.py"

command_full="cd $anode_dir "

for n_nodes in 2 4 6 8 10 12 14 16 20 24 28 32 36 40 48 56 64
do
	echo $n
	
        if [[ "$solver" == "dopri5" ]] || [[ "$solver" == "dopri5_old" ]]
        then
          save_curr="${save}/${solver}_n${n_nodes}_tol${atol}"
        else
          save_curr="${save}/${solver}_n${n_nodes}"
        fi
	
        save_curr="${save_curr}_lr${lr}_bs${batch_size}/$run_id"
        echo $save_curr

        command=" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu python3 $train_file \
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
                 --activation_resblock $activation_resblock\
                 --activation_odeblock $activation_odeblock\
                 --atol $atol\
                 --rtol $rtol\
                 --num_epochs 1"
	command_full="$command_full $command"
done

#command_full="$command_full;"
#echo $command_full

bash -c "$command_full"

### Example: nohup bash scripts/run_dry.sh -g 0  -t configs/training_lr1e-1_bs512.cfg -c configs/solver_tol1e-3.cfg -m configs/model_resnet4_inplanes64.cfg -a configs/normact_bn1-BN_resblock-BNReLU_odeblock-LNReLU.cfg -p configs/pathes_zhores_dry.py -s Euler -r 0&
