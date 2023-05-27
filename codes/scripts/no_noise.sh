title='no_noise1/vit'
num_classes=31
n_epoch=1500
weight_decay=0.00
dropout=0.0
batch_size=96
grad_clip=1
p=1
batchwise_reported=0
activation_layer=relu
pretrained=0
no_noise=0


for optimizer in Adam SGD
do
for lr in 1e-4 1e-5
do
for no_noise in 0 1
do
args="--title $title --device $DEVICE --lr $lr --optimizer $optimizer --num_classes $num_classes --n_epoch $n_epoch --weight_decay $weight_decay --dropout $dropout --batch_size $batch_size --grad_clipping $grad_clip --p $p --batchwise_reported $batchwise_reported --activation $activation_layer --pretrained $pretrained --no_noise $no_noise"
echo begin $args
python -m codes.scripts.no_noise $args
echo end $args
done
done
done
