export http_proxy=
export https_proxy=
n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

export MASTER_ADDR=localhost

# EMA is disabled due to model references between plugins which cannot be maintained in deepcopying
for model in ${MODELS[@]};
do

    if [ $model == sparsified ]; then
        extra="--restricted-affine"
    else
        extra=" "
    fi
    torchrun --nproc_per_node=$n_visible_devices --rdzv_backend=c10d  module_wrapper.py codes.scripts.imagenet1k.imagenet1k   \
        --model $model --start-epoch 0 --epochs 300 --batch-size-per-proc 512 --physical-batch-size 64 --opt adamw --lr 0.003 --wd 0.3 \
        --from-scratch  \
        --data-path 'data/imagenet1k256/ILSVRC/Data/CLS-LOC'    \
        --lr-scheduler cosineannealinglr    \
        --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra    \
        --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
        --amp \
        --log_per_step 100 --physical-epochs 300 \
        --zeroth-bias-clipping 0.1\
        --magic-synapse-rho 0.1 $extra    \
        "$@"
done