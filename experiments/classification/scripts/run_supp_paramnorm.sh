#!/bin/bash

while getopts g:t:c:m:a:i:p:s:r:n:z option
do 
case "${option}"
in
g) gpu=${OPTARG};;
t) training_config=${OPTARG};;
c) solver_config=${OPTARG};;
m) model_config=${OPTARG};;
a) normact_config=${OPTARG};;
i) paramnorm_config=${OPTARG};;
p) path_file=${OPTARG};;
s) solver=${OPTARG};;
r) run_id=${OPTARG};;
n) n_nodes=${OPTARG};;
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

source $paramnorm_config
echo  $param_normalization_resblock $param_normalization_odeblock $param_normalization_bn1

source $path_file
echo $data_root $anode_dir $save_root

num_epochs=350

save="$save_root/classification_bn1-${param_normalization_bn1}${normalization_bn1}_resblock-${param_normalization_resblock}${normalization_resblock}${activation_resblock}_odeblock-${param_normalization_odeblock}${normalization_odeblock}${activation_odeblock}/${network}_inplanes${inplanes}"

train_file="$anode_dir/train.py"

command_full="cd $anode_dir "

        echo $n
	
        if [[ "$solver" == "dopri5" ]] || [[ "$solver" == "dopri5_old" ]]
        then
          save_curr="${save}/${solver}_n${n_nodes}_tol${atol}${rtol}"
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
                 --param_normalization_resblock $param_normalization_resblock\
                 --param_normalization_odeblock $param_normalization_odeblock\
                 --param_normalization_bn1 $param_normalization_bn1\
                 --activation_resblock $activation_resblock\
                 --activation_odeblock $activation_odeblock\
                 --atol $atol\
                 --rtol $rtol\
                 --atol_scheduler '{150: 1.0, 250: 1.0}'\
                 --rtol_scheduler '{150: 1.0, 250: 1.0}'\
                 --num_epochs $num_epochs"
	command_full="$command_full $command"


#command_full="$command_full;"
#echo $command_full

bash -c "$command_full"

### Example: nohup bash scripts/run_supp_paramnorm.sh   -t configs/training_lr1e-1_bs512.cfg -m configs/model_resnet10_inplanes64.cfg -a configs/normact_bn1-NormFree_resblock-NormFreeReLU_odeblock-NormFreeReLU.cfg -i configs/paramnorm_bn1-WN_resblock-WN_odeblock-WN.cfg -p configs/pathes_zhores_supp_paramnorm.py -c configs/solver_tol1e-3.cfg -s Euler -r 0 -g 0 -n 8&
