# BATCH_SIZE=128 bash codes/scripts/imagenet1k/cifar10.sh --title cifar10_lr
# extra="--physical-epochs 75 --fine-grained-checkpints --save-every-epoch 10"
for lr in 1e-3 1e-5;do
    BATCH_SIZE=128 bash codes/scripts/imagenet1k/cifar10.sh --title cifar10_lr_$lr --lr $lr $extra "$@"
done
for batch_size in 64 256; do
    BATCH_SIZE=$batch_size bash codes/scripts/imagenet1k/cifar10.sh --title cifar10_batch_size_$batch_size $extra  "$@"
done
