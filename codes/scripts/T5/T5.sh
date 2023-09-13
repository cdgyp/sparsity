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
    torchrun --nproc_per_node=$n_visible_devices --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:1454  module_wrapper.py codes.scripts.T5.T5  \
        --model_type                t5          \
        --config_name               t5-base     \
        --tokenizer_name            t5-base     \
        --dtype                     bfloat16    \
        --overwrite_output_dir                  \
        --do_train                              \
        --do_eval                               \
        --per_device_train_batch_size   512     \
        --per_device_eval_batch_size    64      \
        --gradient_accumulated_steps    8       \
        --max_steps                     1e5     \
        --learning_rate                 0.005   \
        --weight_decay                  0.001   \
        --warmup_steps                  2000    \
        --logging_steps                 200     \
        --save_steps                    5000    \
        --eval_steps                    2500    \
        --from_disk                             \
        --dataset_name              'data/c4'   \
        --max_seq_length                512     \
        --preprocessing_num_workers     8       \
        --compile                               \
        $extra                                  \
        "$@"
done