for args in "" "--do-filtering --filter-threshold 0.95" "--do-filtering --filter-threshold -0.95"; do
    python -m codes.scripts.adversarial.covering --dataset data/imagenet1k256/ILSVRC/Data/CLS-LOC/val --title imagenet1k/$EPS/ --epsilon $EPS --batch-size 32 --n-samples 64000 $args "$@"
done