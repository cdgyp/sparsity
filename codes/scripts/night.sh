num_classes=10
n_epoch=10
weight_decay=0.0
dropout=0

for lr in 3e-2 1e-2 1e-3 1e-4 1e-5 1e-6
do
for optimizer in Adam SGD RMSprop
do
echo begin lr=$lr optimizer=$optimizer
python -m codes.scripts.vit --device $DEVICE --lr $lr --optimizer $optimizer --num_classes $num_classes --n_epoch $n_epoch --weight_decay $weight_decay --dropout $dropout
echo end lr=$lr optimizer=$optimizer
done
done