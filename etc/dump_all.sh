for dx in +1.6 -1.6;do
for dy in +1.6 -1.6; do

idx=$(echo "0$dx" | bc)
idy=$(echo "0$dy" | bc)

python etc/dump.py --source-dir runs/vit/manipulate_width_implicit_relu2/vit/DB-MLP_only_image_size224/batch_size64/physical_batch_size64/dropout0.0/lr0.0001/optimizerAdam/num_classes10/weight_decay0.0/n_epoch100/grad_clipping1.0/p1.0/batchwise_reported0/warmup_epoch5/initial_lr_ratio0.1/activation_layerjumping_weird_relu2/pretrained0/log_per_step10/half_interval1.5/shift_x$idx/shift_y$idy/alpha_x1.0/alpha_y1.0/careful_bias_initialization0/rezero0/zeroth_bias1/datasetcifar10/mixed_precisionFalse/resumeNone_/*/activation_concentration_train/_obs_pre_activation/ --output-dir dumps/validations/dx=$dx,dy=$dy/derivative_sparsity/

python etc/dump.py --source-dir runs/vit/manipulate_width_implicit_relu2/vit/DB-MLP_only_image_size224/batch_size64/physical_batch_size64/dropout0.0/lr0.0001/optimizerAdam/num_classes10/weight_decay0.0/n_epoch100/grad_clipping1.0/p1.0/batchwise_reported0/warmup_epoch5/initial_lr_ratio0.1/activation_layerjumping_weird_relu2/pretrained0/log_per_step10/half_interval1.5/shift_x$idx/shift_y$idy/alpha_x1.0/alpha_y1.0/careful_bias_initialization0/rezero0/zeroth_bias1/datasetcifar10/mixed_precisionFalse/resumeNone_/*/derivative2 --output-dir dumps/validations/dx=$dx,dy=$dy/derivative_norm/

done
done

exit

for type in test train; do
for model_type in vanilla sparsified; do
    dir=runs/imagenet1k/from_scratch3/${model_type}/*/activation_concentration_?${type}?_?obs?activation
    python etc/dump.py --source-dir $dir --output-dir dumps/layer_norm_ablation/${model_type}/activation_concentration_$type
done
done

exit

for exp in runs/imagenet1k/cifar10*/; do
    for model_type in vanilla sparsified; do
        trial=$(ls $exp/$model_type | tail -n1) # the last experiment
        echo $trial
        for type in test train; do
            python etc/dump.py --source-dir $exp/$model_type/$trial/activation_concentration_$type/?obs?activation --output-dir dumps/cifar10/$(basename $exp)/$model_type/activation_concentration_$type
        done
    done
done

exit



# Experiments for Productivity

for full_exp_type in "T5/finetuning from_scratch" "imagenet1k/from_scratch5 finetuning_hard_uplifting";
do
    IFS='/' read -ra parts <<< "$full_exp_type"
    task="${parts[0]}"
    trials="${parts[1]}"
    for trial in $trials;
    do
        if [ $trial == from_scratch5 ]; then
            output_trial=from_scratch
        elif [ $trial == finetuning_hard_uplifting ]; then
            output_trial=finetuning
        else
            output_trial=$trial
        fi
        for model_type in vanilla sparsified;
        do
            for activation_type in activation pre_activation;
            do
                for stage_type in train test;
                do
                    if [[ "$task/$trial/$stage_type" == "T5/from_scratch/test" ]]; then
                        dir=runs/$task/reeval/$model_type/*/activation_concentration_*${stage_type}*/?obs?$activation_type
                        echo "!!!$dir -> dumps/$task/$output_trial/$model_type/$activation_type"
                    else
                        dir=runs/$task/$trial/$model_type/*/activation_concentration_*${stage_type}*/?obs?$activation_type
                    fi
                    python etc/dump.py --source-dir $dir --output-dir dumps/$task/$output_trial/$model_type/$activation_type
                done
            done

            if [ $task == imagenet1k ]; then
                for stage_type in "" test_;
                do
                    for top_type in 1 5;
                    do
                        python etc/dump.py --source-dir runs/$task/$trial/$model_type/*/${stage_type}acc/ --output-dir dumps/$task/$output_trial/$model_type/
                    done
                done
            else
                python etc/dump.py --source-dir runs/$task/$trial/$model_type/*/eval_metrics/ --output-dir dumps/$task/$output_trial/$model_type/
            fi

            for matrix_type in kkT M;
            do

                python etc/dump.py --source-dir runs/$task/$trial/$model_type/*/spectral_increase/?obs?$matrix_type --output-dir dumps/$task/$output_trial/$model_type/spectral_increase/$matrix_type

            done
            python etc/dump.py --source-dir runs/$task/$trial/$model_type/*/norm_g_V --output-dir dumps/$task/$output_trial/$model_type/norm_g_V

            python etc/dump.py --source-dir runs/$task/$trial/$model_type/ --output-dir dumps/$task/$output_trial/$model_type/etc/
        done
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
