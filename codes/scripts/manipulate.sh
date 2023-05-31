title='manipulate_rezero/vit'
num_classes=10
n_epoch=100
weight_decay=0.00
dropout=0.0
batch_size=64
grad_clip=1
p=1
batchwise_reported=0
activation_layer=weird
pretrained=0
half_interval=0.5
shift_x=0
shift_y=0
rezero=1
careful_bias_initialization=0

for optimizer in Adam SGD
do
for lr in 3e-4 1e-5
do
for shift_x in 0.6 -0.6
do
for shift_y in 0.6 -0.6
do
args="--title $title --device $DEVICE --lr $lr --optimizer $optimizer --num_classes $num_classes --n_epoch $n_epoch --weight_decay $weight_decay --dropout $dropout --batch_size $batch_size --grad_clipping $grad_clip --p $p --batchwise_reported $batchwise_reported --activation $activation_layer --pretrained $pretrained --half_interval $half_interval --shift_x $shift_x --shift_y $shift_y --rezero $rezero --careful_bias_initialization $careful_bias_initialization"
echo begin $args
python -m codes.scripts.manipulate $args
echo end $args
done
done
done
done
