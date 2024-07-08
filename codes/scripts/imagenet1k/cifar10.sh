export http_proxy=
export https_proxy=
batch_size=$BATCH_SIZE
n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
batch_size_per_proc=$((batch_size / n_visible_devices))
echo $batch_size_per_proc samples per process

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
        --model $model --start-epoch 0 --epochs 150 --batch-size-per-proc $batch_size_per_proc --physical-batch-size 64 --opt adamw --lr 0.0001 --wd 0.05 \
        --from-scratch  \
        --data-path 'data/cifar10'    \
        --lr-scheduler cosineannealinglr    \
        --label-smoothing 0.11 \
        --clip-grad-norm 1 --ra-sampler \
        --amp \
        --log_per_step 25 --save-every-epoch 10 --physical-epochs 150 \
        --zeroth-bias-clipping 0.1\
        --fine-grained-checkpoints \
	--mixup-alpha 0.2 	\
	--cutmix-alpha 1.0      \
        $extra    \
        "$@"
done

--auto-augment ra    \
