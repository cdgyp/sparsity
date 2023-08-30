partition_1="    100     300     500    "
partition_2="        200     400"

partition_name=partition_$PARTITION

for n in ${!partition_name};
do
python -m codes.scripts.marchenko_pastur.mnist --device $DEVICE --dim-hidden $n --n-parallel-models 5 --training --centralized "$@"
done
