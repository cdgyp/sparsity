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
