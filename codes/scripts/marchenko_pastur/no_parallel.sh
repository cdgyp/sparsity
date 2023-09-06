threshold=1e-3

partition_1="336    128  624"
partition_2="400    224      512"

partition_2_1="1024      800       16"
partition_2_2="912    720       64"

wd_partition_1="0.1 0.01"
wd_partition_2="0.001 0.0"

session_name="marchenko-pastur-$PARTITION-$WD_PARTITION"

partition_name=partition_$PARTITION
wd_partition_name=wd_partition_$WD_PARTITION


extra="$@"


for n in ${!partition_name};
do
log_dir="runs/marchenko_pastur/$TITLE/wd$wd/$n"
if [ -d "$log_dir" ]; then
    echo $log_dir previously done
else
    args="--dim-hidden $n --training --n-parallel-models 1 --centralized --threshold $threshold --no-affine --n-epochs 100 --weight-decay $wd --title $TITLE $extra"
    python -m codes.scripts.marchenko_pastur.mnist --device $DEVICE $args
fi
done
