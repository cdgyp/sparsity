export http_proxy=
export https_proxy=

# EMA is disabled due to model references between plugins which cannot be maintained in deepcopying
for model in ${MODELS[@]};
do

    if [ $model == sparsified ]; then
        extra="--restricted-affine"
    else
        extra=""
    fi
    python -m codes.scripts.imagenet1k.imagenet1k   \
        --model $model --start-epoch 0 --epochs 300 --batch-size 512 --physical-batch-size 64 --opt adamw --lr 0.003 --wd 0.3 \
        --from-scratch  \
        --data-path 'data/imagenet1k256/ILSVRC/Data/CLS-LOC'    \
        --lr-scheduler cosineannealinglr    \
        --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra    \
        --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
        --amp \
        --device $DEVICE  --log_per_step 100 --physical-epochs 300 \
        --zeroth-bias-clipping 0.1\
        --magic-synapse-rho 0.1\
        "$extra"    \
        "$@"
done