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
        --do_train                      false   \
        --do_eval                               \
        --max_obs_batch_size            32      \
        --per_device_eval_batch_size    32      \
        --max_steps                     0       \
        --learning_rate                 1e-9       \
        --weight_decay                  0       \
        --logging_steps                 25      \
        --eval_steps                    1       \
        --from_disk                             \
        --dataset_name              'data/c4'   \
        --max_seq_length                512     \
        --warmup_steps                  1       \
        --scan_eval                             \
        --dir_to_checkpoints        runs/T5/from_scratch/$model/*/save  \
        $extra                                  \
        "$@"
done