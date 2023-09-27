export http_proxy=
export https_proxy=

batch_size=2048
n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
batch_size_per_proc=$((batch_size / n_visible_devices))
echo $batch_size_per_proc samples per process

# EMA is disabled due to model references between plugins which cannot be maintained in deepcopying
IFS=','
for model in $MODELS;
do
    torchrun --nproc_per_node=$n_visible_devices module_wrapper.py codes.scripts.imagenet1k.imagenet1k   \
        --model $model --start-epoch 0 --epochs 15 --batch-size $batch_size_per_proc --physical-batch-size 64 --opt adamw --lr 0.003 --wd 0.3 \
        --data-path 'data/imagenet1k256/ILSVRC/Data/CLS-LOC'    \
        --lr-scheduler cosineannealinglr  \
        --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra    \
        --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
	--workers 4 \
        --amp \
        --log_per_step 100 --physical-epochs 15 \
        --dont-resume-lr-schedulers \
        --finetune runs/finetuning/start/start.pth \
        --lora --lora-r 16  \
        --activation-mixing-epoch 5 \
        "$@"
done
