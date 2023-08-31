title='manipulate_width_implicit_relu2/vit'
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
half_interval=1.5
shift_x=0
shift_y=0
alpha_x=1
alpha_y=1
implicit_adversarial_samples=1
rezero=0
careful_bias_initialization=0

for optimizer in Adam
do
for lr in 1e-4 1e-5
do
for shift_x in 1.6 -1.6
do
for shift_y in 1.6 -1.6
do
args="--title $title --device $DEVICE --lr $lr --optimizer $optimizer --num_classes $num_classes --n_epoch $n_epoch --weight_decay $weight_decay --dropout $dropout --batch_size $batch_size --grad_clipping $grad_clip --p $p --batchwise_reported $batchwise_reported --activation $activation_layer --pretrained $pretrained --half_interval $half_interval --shift_x $shift_x --shift_y $shift_y --rezero $rezero --careful_bias_initialization $careful_bias_initialization --implicit_adversarial_samples $implicit_adversarial_samples --alpha_x $alpha_x --alpha_y $alpha_y"
echo begin $args
python -m codes.scripts.manipulate $args
echo end $args
done
done
done
done
