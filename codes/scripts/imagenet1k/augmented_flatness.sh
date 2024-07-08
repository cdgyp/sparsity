export http_proxy=
export https_proxy=

batch_size=2048
n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
batch_size_per_proc=$((batch_size / n_visible_devices))
echo $batch_size_per_proc samples per process

dir=/data/pz/sparsity/runs/imagenet1k/from_scratch5/sparsified/20230915-133443/save/

# EMA is disabled due to model references between plugins which cannot be maintained in deepcopying
IFS=','
for model in $MODELS;
do
        for checkpoint in $dir/*; do
        torchrun --master_port=2333 --nproc_per_node=$n_visible_devices module_wrapper.py codes.scripts.imagenet1k.imagenet1k   \
                --model $model --start-epoch 0 --epochs 300 --batch-size $batch_size_per_proc --physical-batch-size 128 --opt adamw --lr 0.0 --wd 0.0 \
                --data-path 'data/imagenet1k256/ILSVRC/Data/CLS-LOC'    \
                --lr-scheduler cosineannealinglr  \
                --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra    \
                --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
	        --workers 4 \
                --amp \
                --log_per_step 1 --physical-epochs 300 \
                --dont-resume-lr-schedulers \
                --resume $checkpoint \
                --max-iteration 128 \
                --post-training-only \
                --augmented-flatness-only \
                --title augmented_flatness_128_allepochs_sparsified \
                --lora-r 0 \
                --no-testing \
                --print-freq 5 \
                "$@"
        done
done
