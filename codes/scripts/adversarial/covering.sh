for full_type in "row_filtering/--do-filtering --filter-threshold 0.95" "no_filtering/" "column_filtering/--do-filtering --filter-threshold -0.95"; do
    IFS='/' read -ra parts <<< "$full_type"
    task_name="${parts[0]}"
    args="${parts[1]}"
    echo python -m codes.scripts.adversarial.covering --dataset data/imagenet1k256/ILSVRC/Data/CLS-LOC/val --title imagenet1k/$task_name --epsilon $EPS --batch-size 16 --n-samples 64000 $args "$@"
    python -m codes.scripts.adversarial.covering --dataset data/imagenet1k256/ILSVRC/Data/CLS-LOC/val --title imagenet1k/$task_name --epsilon $EPS --batch-size 16 --n-samples 64000 $args "$@"
done