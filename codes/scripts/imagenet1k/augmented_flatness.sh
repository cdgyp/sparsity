export http_proxy=
export https_proxy=

batch_size=128
n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
batch_size_per_proc=$((batch_size / n_visible_devices))
echo $batch_size_per_proc samples per process

# EMA is disabled due to model references between plugins which cannot be maintained in deepcopying
IFS=','
for model in $MODELS;
do
        for checkpoint in model_0.pth model_25.pth model_50.pth model_100.pth model_101.pth model_150.pth model_200.pth model_250.pth model_280.pth model_290.pth checkpoint.pth; do
        torchrun --nproc_per_node=$n_visible_devices module_wrapper.py codes.scripts.imagenet1k.imagenet1k   \
                --model $model --start-epoch 0 --epochs 300 --batch-size $batch_size_per_proc --physical-batch-size 64 --opt adamw --lr 0.0 --wd 0.0 \
                --data-path 'data/imagenet1k256/ILSVRC/Data/CLS-LOC'    \
                --lr-scheduler cosineannealinglr  \
                --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra    \
                --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
	        --workers 4 \
                --amp \
                --log_per_step 1 --physical-epochs 300 \
                --dont-resume-lr-schedulers \
                --resume runs/imagenet1k/from_scratch5/$model/*/save/$checkpoint \
                --max-iteration 1 \
                --post-training-only \
                --augmented-flatness-only \
                --title augmented_flatness \
                --lora-r 0 \
                --no-testing \
                --print-freq 100 \
                "$@"
        done
done
