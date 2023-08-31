threshold=1e-6

partition_1="16    128      336       512    "
partition_2="   64     224      400"

partition_2_1="624      800       1024"
partition_2_2="    720       912"

partition_name=partition_$PARTITION

for wd in 0.1 0.01 0.001 0.0;
do
for n in ${!partition_name};
do
python -m codes.scripts.marchenko_pastur.mnist --device $DEVICE --dim-hidden $n --training --n-parallel-models 1 --centralized --weight-decay $wd --threshold $threshold "$@"
done
done
1