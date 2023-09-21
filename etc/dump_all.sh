# Experiments for Productivity

for model_type in vanilla sparsified;
do
    for activation_type in activation pre_activation;
    do
        for stage_type in train test;
        do
            python etc/dump.py --source-dir runs/imagenet1k/from_scratch5/$model_type/*/activation_concentration_\($stage_type\)/\[obs\]$activation_type --output-dir dumps/imagenet1k/$model_type/$activation_type
        done
    done

    for stage_type in "" test_;
    do
        for top_type in 1 5;
        do
            python etc/dump.py --source-dir runs/imagenet1k/from_scratch5/$model_type/*/${stage_type}acc/ --output-dir dumps/imagenet1k/$model_type/
        done
    done

    for matrix_type in kkT M;
    do

        python etc/dump.py --source-dir runs/imagenet1k/from_scratch5/$model_type/*/spectral_increase/\[obs\]$matrix_type --output-dir dumps/imagenet1k/$model_type/spectral_increase

    done
done

exit


## Finetuning

for type in vanilla sparsified;
do
    python etc/dump.py --source-dir runs/imagenet1k/finetune/$type/*/activation_concentration_\(test\)_\[obs\]activation/ --output-dir dumps/finetuning/$type/
done

root_cifar10_runs="runs/improvements/show"
for record in $(ls $root_cifar10_runs | grep "="); do
    num_files=$(ls -l $root_cifar10_runs/$record/ | grep "20230607-144128" | wc -l)
    if [ $num_files -gt 0 ]
    then
        path=$record/20230607-144128
    else
        path=$record
    fi
    python etc/dump.py --source-dir $root_cifar10_runs/$path/pseudo_sparsity_\[obs\]activation --output-dir dumps/cifar10/$record
done


for dx in +1.6 -1.6;do
for dy in +1.6 -1.6; do

idx=$(echo "0$dx" | bc)
idy=$(echo "0$dy" | bc)

python etc/dump.py --source-dir runs/manipulate_width_implicit_relu2/vit_\(activation_layer\=weird_relu2/alpha_x\=1.0/alpha_y\=1.0/batch_size\=64/batchwise_reported\=0/careful_bias_initialization\=0/dataset\=cifar10/dropout\=0.0/grad_clipping\=1.0/half_interval\=1.5/image_size\=224/implicit_adversarial_samples\=1/initial_lr_ratio\=0.1/log_per_step\=10/lr\=0.0001/n_epoch\=100/num_classes\=10/optimizer\=Adam/p\=1.0/pretrained\=0/rezero\=0/shift_x\=$idx/shift_y\=$idy/warmup_epoch\=5/weight_decay\=0.0\)/*/pseudo_sparsity_\[obs\]activation/ --output-dir dumps/validations/dx=$dx,dy=$dy/

done
done


for full in finetune/vanilla/20230810-224053 from_scratch4/sparsified/20230809-160718; do
for matrix_type in kkT M hadamard;do

type=$(echo $full | sed -E "s/.*(vanilla|sparsified).*/\1/g")
folder=finetuning
if [ $type == sparsified ]; then
    folder=imagenet1k
fi

python etc/dump.py --source-dir "runs/imagenet1k/$full/verification_norm1_${matrix_type}_[obs]diagonals/" --output-dir dumps/$folder/$type/spectral/

done
done


python etc/dump.py --source-dir "runs/imagenet1k/from_scratch4/sparsified/20230809-160718/activation_concentration_(train)_[obs]activation/" --output-dir dumps/imagenet1k/sparsified/from_scratch4/ --filter-dname '4' '7' '8' '10'

bash etc/dump_mp.sh
