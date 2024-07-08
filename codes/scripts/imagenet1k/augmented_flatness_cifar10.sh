export http_proxy=
export https_proxy=

batch_size=128
n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
batch_size_per_proc=$((batch_size / n_visible_devices))
echo $batch_size_per_proc samples per process

dir=/home/pz/sparsity/runs/imagenet1k/cifar10/vanilla/20240428-155216/save/

# EMA is disabled due to model references between plugins which cannot be maintained in deepcopying
IFS=','
for model in $MODELS;
do
        for checkpoint in $dir/*; do
        torchrun --master_port=2333 --nproc_per_node=$n_visible_devices module_wrapper.py codes.scripts.imagenet1k.imagenet1k   \
                --model $model --start-epoch 0 --epochs 300 --batch-size $batch_size_per_proc --physical-batch-size 128 --opt adamw --lr 0.0 --wd 0.0 \
                --data-path 'data/cifar10'    \
                --lr-scheduler cosineannealinglr  \
                --label-smoothing 0.11 --mixup-alpha 0.2 \
                --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
	        --workers 4 \
                --from-scratch \
                --amp \
                --log_per_step 1 --physical-epochs 300 \
                --dont-resume-lr-schedulers \
                --resume $checkpoint \
                --post-training-only \
                --augmented-flatness-only \
                        --correct-samples-only \
                --title augmented_flatness_cifar10_allepochs_correct_only \
                --lora-r 0 \
                --no-testing \
                --print-freq 5 \
                "$@"
        done
done
