export http_proxy=
export https_proxy=

n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

export MASTER_ADDR=localhost


for model in ${MODELS[@]};
do

    if [ $model == sparsified ]; then
        extra="--restricted_affine --db_mlp --zeroth_bias_clipping 0.1 --jsrelu"
    else
        extra=" "
    fi
    torchrun --nproc_per_node $n_visible_devices --rdzv_backend=c10d module_wrapper.py codes.scripts.T5.T5  \
        --model_type                t5                      \
        --config_name               hf_caches/t5-base       \
        --tokenizer_name            hf_caches/t5-base       \
        --dtype                     float32     \
        --overwrite_output_dir                  \
        --do_train                              \
        --do_eval                               \
        --per_device_train_batch_size_at_usual   64      \
        --gradient_accumulated_steps_at_usual    1       \
        --per_device_train_batch_size_at_logging   16      \
        --gradient_accumulated_steps_at_logging    4       \
        --per_device_eval_batch_size    64      \
        --max_steps                     1e5     \
        --learning_rate                 0.01    \
        --weight_decay                  0.001   \
        --warmup_steps                  10000   \
        --logging_steps                 50     \
        --save_steps                    5000    \
        --eval_steps                    2000    \
        --from_disk                             \
        --dataset_name              'data/c4'   \
        --max_seq_length                512     \
        $extra                                  \
        "$@"
done