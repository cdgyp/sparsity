title='improvements/vit'
num_classes=10
n_epoch=100
weight_decay=1e-4
dropout=0.2
batch_size=64
grad_clip=1
p=1
batchwise_reported=0
pretrained=0
half_interval=1.5
shift_x=0
shift_y=0
alpha_x=1
alpha_y=1
rezero=0
careful_bias_initialization=0

shift_x=0
shift_y=0
half_interval=0

for optimizer in AdamW
do
for lr in 1e-4
do
for improvements in '--activation_layer jumping_squared_relu' '--activation_layer relu'  '--activation_layer jumping_squared_relu --zeroth-bias 1' '--activation_layer relu --zeroth-bias 1' # 
do
args="--title $title --lr $lr --optimizer $optimizer --num_classes $num_classes --n_epoch $n_epoch --weight_decay $weight_decay --dropout $dropout --batch_size $batch_size --grad_clipping $grad_clip --p $p --batchwise_reported $batchwise_reported  --pretrained $pretrained --half_interval $half_interval --shift_x $shift_x --shift_y $shift_y --rezero $rezero --careful_bias_initialization $careful_bias_initialization  --alpha_x $alpha_x --alpha_y $alpha_y $improvements --mixed_precision $@"
echo begin $args
python -m codes.scripts.manipulate $args
echo end $args
done
done
done
