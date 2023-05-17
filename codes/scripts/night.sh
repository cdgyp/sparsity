title='vit6/vit'
num_classes=10
n_epoch=100
weight_decay=0.00
dropout=0.0
batch_size=96
grad_clip=1
p=1
batchwise_reported=0

for optimizer in Adam SGD
do
for lr in 1e-4 1e-5
do
args="--title $title --device $DEVICE --lr $lr --optimizer $optimizer --num_classes $num_classes --n_epoch $n_epoch --weight_decay $weight_decay --dropout $dropout --batch_size $batch_size --grad_clipping $grad_clip --p $p --batchwise_reported $batchwise_reported"
echo begin $args
python -m codes.scripts.vit2 $args
echo end $args
done
done
