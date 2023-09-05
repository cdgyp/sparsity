threshold=1e-3


session_name="marchenko-pastur-$(echo $WD | sed -E 's/\.//')"

extra="$@"

tmux new-session -d -s $session_name

for PARTITION in 1 2 2_1 2_2;
do
tmux send-keys -t $session_name "DEVICE=$DEVICE PARTITION=$PARTITION bash codes/scripts/marchenko_pastur/no_parallel.sh --weight-decay $WD $extra" C-m
tmux new-window -t $session_name
done

