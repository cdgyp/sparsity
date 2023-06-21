export http_proxy=
export https_proxy=

# EMA is disabled due to model references between plugins which cannot be maintained in deepcopying
for model in vanilla sparsified;
do
    python -m codes.scripts.imagenet1k.imagenet1k   \
        --model $model --start-epoch 300 --epochs 310 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3 \
        --data-path 'data/imagenet1k256/ILSVRC/Data/CLS-LOC'    \
        --lr-scheduler cosineannealinglr    \
        --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra    \
        --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
        --device $DEVICE
done