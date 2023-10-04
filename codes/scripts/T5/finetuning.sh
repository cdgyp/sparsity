export http_proxy=
export https_proxy=

n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

export MASTER_ADDR=localhost

batch_size=256
gradient_accumulated_steps=2
n_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
batch_size_per_proc=$((batch_size / n_visible_devices / gradient_accumulated_steps))
echo $batch_size_per_proc samples per process

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
        --per_device_train_batch_size   $batch_size_per_proc    \
        --gradient_accumulated_steps    $gradient_accumulated_steps \
        --max_obs_batch_size            16      \
        --per_device_eval_batch_size    16      \
        --max_steps                     1e4     \
        --learning_rate                 0.01    \
        --weight_decay                  0.001   \
        --warmup_steps                  1000    \
        --logging_steps                 50      \
        --save_steps                    1000    \
        --eval_steps                    2500    \
        --from_disk                             \
        --dataset_name              'data/c4'   \
        --max_seq_length                512     \
        --gradient_checkpointing                \
        --mixed_activation                      \
        --activation_mixing_steps       3000    \
        --layernorm_uplifting_steps     3000    \
        --finetune                      start_points/t5-base  \
        --lora_r                        192     \
        $extra                                  \
        "$@"
done