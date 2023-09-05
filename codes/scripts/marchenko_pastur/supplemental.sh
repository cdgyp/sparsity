threshold=1e-6

session_name="marchenko-pastur-supplemental"
extra="$@"


tmux new-session -d -s $session_name
for full in 0.1/800 0.01/128 0.01/400 0.01/800;
do

IFS='/' read -ra parts <<< "$full"
wd="${parts[0]}"
n="${parts[1]}"

tmux send-keys -t $session_name "python -m codes.scripts.marchenko_pastur.mnist --device $DEVICE --dim-hidden $n --training --n-parallel-models 1 --centralized --weight-decay $wd --threshold $threshold $extra" C-m
tmux new-window -t $session_name

done