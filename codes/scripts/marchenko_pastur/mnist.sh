threshold=1e-6

partition_1="16    128      336       512    "
partition_2="   64     224      400"

partition_2_1="624      800       1024"
partition_2_2="    720       912"

wd_partition_1="0.1 0.01"
wd_partition_2="0.001 0.0"

session_name="marchenko-pastur-$PARTITION-$WD_PARTITION"

partition_name=partition_$PARTITION
wd_partition_name=wd_partition_$WD_PARTITION


extra="$@"

tmux new-session -d -s $session_name

for wd in ${!wd_partition_name};
do
for n in ${!partition_name};
do
args="--dim-hidden $n --training --n-parallel-models 1 --centralized --weight-decay $wd --threshold $threshold --no-affine $extra"
tmux send-keys -t $session_name "python -m codes.scripts.marchenko_pastur.mnist --device $DEVICE $args" C-m
tmux new-window -t $session_name
done
done

