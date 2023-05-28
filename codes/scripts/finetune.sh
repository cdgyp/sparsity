title='vit-finetune/vit'
num_classes=10
n_epoch=20
weight_decay=0.00
dropout=0.0
batch_size=96
grad_clip=1
p=1
batchwise_reported=0
activation_layer=relu
pretrained=1

for optimizer in Adam SGD
do
for lr in 1e-4 1e-5
do
for activation_layer in relu srelu leaky_relu weird_leaky_relu
do
args="--title $title --device $DEVICE --lr $lr --optimizer $optimizer --num_classes $num_classes --n_epoch $n_epoch --weight_decay $weight_decay --dropout $dropout --batch_size $batch_size --grad_clipping $grad_clip --p $p --batchwise_reported $batchwise_reported --activation $activation_layer --pretrained $pretrained"
echo begin $args
python -m codes.scripts.vit2 $args
echo end $args
done
done
done
