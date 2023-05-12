num_classes=30
n_epoch=50

for lr in 1e-3 1e-4 1e-5 1e-6
do
for optimizer in SGD Adam RMSprop
do
echo begin lr=$lr optimizer=$optimizer
python -m codes.scripts.vit --device $DEVICE --lr $lr --optimizer $optimizer --num_classes $num_classes --n_epoch $n_epoch
echo end lr=$lr optimizer=$optimizer
done
done