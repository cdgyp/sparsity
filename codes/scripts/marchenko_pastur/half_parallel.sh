threshold=1e-3


session_name="marchenko-pastur-$PARTITION"

extra="$@"

tmux new-session -d -s $session_name

for wd in 0.0 0.1 0.01;
do
tmux send-keys -t $session_name "DEVICE=$DEVICE PARTITION=$PARTITION bash codes/scripts/marchenko_pastur/no_parallel.sh --weight-decay $wd $extra" C-m
tmux new-window -t $session_name
done

