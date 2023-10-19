# Experiments in "A Theoretical Explanation of Activation Sparsity through Flat Minima and Adversarial Robustness"

## Initialization

Use the following commands to pull submodules required by this repo:
```bash
git submodule update --init
```

## File Structure

```bash
.
├── codes                                   # codes for experiments
│   ├── base                                # codes to ease experiments
│   ├── modules                             # task agnostic codes
│   │   ├── activations.py                  # activation functions, including J-SquaredReLU
│   │   ├── hooks.py                        # hooks used to track sparsities, spectral increase, etc.
│   │   ├── jsrelu_ext                      # C++ extension (to reduce overhead) of J-SquaredReLU
│   │   ├── lora.py                         # LoRA implementation
│   │   ├── magic.py                        # tokenwise synapse noise, not used in experiments
│   │   ├── relu_vit.py                     # ViT adopted from TorchVision but with customized activation functions
│   │   ├── robustness.py                   # Zeroth Bias, DB-MLP and restricted LayerNorm
│   │   ├── sparsify.py                     # putting all sparsification methods together for easier use
│   │   └── vit.py                          # ViT adopted from `pytorch-vit`, deprecated
│   ├── scheduler                           
│   │   └── sine.py                         # SineAnnealing scheduler, previously designed for GELU adaptation in finetuning, deprecated
│   └── scripts                             # task specific codes
│       ├── after_training                  # eigenvalue decomposition to demonstrate spectral concentration
│       ├── finetune.sh                     # deprecated
│       ├── imagenet1k                      # codes for training and finetuning ViT-Base on ImageNet-1k
│       ├── T5                              # codes for training and finetuning T5 on C4
│       ├── improvements_imagenet1k.sh      # deprecated
│       ├── improvements.sh                 # deprecated
│       ├── manipulate.py                   # codes of experiments for validation, i.e., manipulating activation sparsity through gradient sparsity
│       ├── manipulate.sh                   # script to launch experiments for validation
│       ├── marchenko_pastur                # codes for verifying the applicability (Sec 8.4) of spectral concentration bounds (Theorem 8)
│       ├── mlp.py                          # deprecated
│       ├── night.sh                        # deprecated
│       ├── no_noise.py                     # deprecated
│       ├── no_noise.sh                     # deprecated
│       ├── vit2.py                         # deprecated
│       └── vit.py                          # deprecated
├── data                                    # directory to hold datasets
│   ├── c4
│   ├── imagenet1k256 
│   └── mnist
├── runs                                    # directory to hold experiments logs and checkpoints
├── dumps                                   # .csv version of logs extracted from TensorBoard databases
├── etc                                     # codes to dump TensorBoard logs in order to produce dumps/
├── extensions                              # where J-SquaredReLU C++ extension is installed
├── install_extensions.sh                   # script to install J-SquaredReLU C++ extension
├── hf_caches                               # manually cloned Huggingface repos
│   └── t5-base
├── clone_T5_repo.sh                        # script to clone Huggingface repos
├── module_wrapper.py                       # wrapping python modules because we use relative imports
└── start_points                            # folder to hold checkpoints from which finetuning starts
    └── t5-base
```

## Dependencies

See `./requirements.txt`.

Note we require newest `torchvsion` to use ViT provided by it.

## Dataset Preparation

Download MNIST and ImageNet-1k and place under `data/` so that the directory looks like

```bash
├── c4
├── imagenet1k256
│   └── ILSVRC
│       └── Data
│           └── CLS-LOC
│               ├── test
│               ├── train
│               └── val
└── mnist
    └── MNIST
        └── raw
            ├── t10k-images-idx3-ubyte.gz
            ├── t10k-labels-idx1-ubyte.gz
            ├── train-images-idx3-ubyte.gz
            └── train-labels-idx1-ubyte.gz
```

Before downloading ImageNet-1k we resized images to 256 x 256 online in order to save downloading time and storage.

Run the following commands to download a subset of C4:

```bash
bash codes/scripts/T5/c4.sh
```

After it finishes `data/` should look like

```bash
.
├── c4
│   ├── train_part_0
│   ├── train_part_1
│   ├── train_part_2
│   ├── ...
│   ├── validation_part_0
│   ├── validation_part_1
│   ├── validation_part_2
│   └── validation_part_3
...
```

## Reproducing Experiments

All commands to reproduce experiments should be executed at the root of this repo.

Raw data of our runs can be found on [Huggingface](https://huggingface.co/pgze/sparsified_transformers/tree/main), which is similarly structured as `runs/`.
The terminology in experiment tracking can be different with that in the manuscript. For example, "concentration" instead of sparsity is used during experiments.

GPU can be specified by environmental variable `CUDA_VISIBLE_DEVICES` in all experiments.

### Experiments for Validation (Sec 6)

Run the following codes:

```
bash codes/scripts/manipulate.sh
```

When the experiments are finished, results can be found at `runs/manipulate_width_implicit_relu2` and viewed by opening TensorBoard at that directory. Filter by `shift_x` and `shift_y` to see different activation concentration manipulated by gradient sparsity.

### ViT-Base/16 on ImageNet-1k (Sec 7)

To train from scratch, use the following commands:

```bash
MODELS=sparsified codes/scripts/imagenet1k/imagenet1k.sh --title from_scratch
MODELS=vanilla codes/scripts/imagenet1k/imagenet1k.sh --title from_scratch
```

The results can be found under `runs/imagenet1k/from_scratch/`, where `sparsified` and `vanilla` hold results of modified and vanilla models.

To finetune from sparsity, first select a checkpoint from vanilla training. For example, `runs/imagenet1k/from_scratch/vanilla/*/save/checkpoint.pth` is the newest checkpoint. Copy it to `runs/finetuning/start/start.pth`. Then run the following commands:
```bash
MODELS=sparsified codes/scripts/imagenet1k/finetuning.sh --title finetuning
MODELS=vanilla codes/scripts/imagenet1k/finetuning.sh --title finetuning
```
The results can be found under `runs/imagenet1k/finetuning/`.

Feel free to adjust hyperparameters and parameters of logging and checkpointing in the above scripts.

### T5-Base on C4 (Sec 7)

To train from scratch, run
```bash
MODELS=sparsified codes/scripts/T5/T5.sh --title from_scratch/sparsified
MODELS=vanilla codes/scripts/T5/T5.sh --title from_scratch/vanilla
```
The results can be found under `runs/T5/from_scratch/`.

Before finetuning, copy a checkpoint from vanilla training. For example, copy or link `runs/T5/from_scratch/*/save/checkpoint-100000/pytorch_model.bin` to `start_points/t5-base/pytorch_model.bin`. Then run
```bash
MODELS=sparsified codes/scripts/T5/finetuning.sh --title finetuning/sparsified
MODELS=vanilla codes/scripts/T5/finetuning.sh --title finetuning/vanilla
```
The results can be found under `runs/T5/finetuning/`.

### Applicability of Theorem 8 (Sec 8.4)

Since models are small but there are too many of them, the scripts of this experiments create a lot of parallel training processes. So be careful with the burden and adjust how trainings are paralleled in scripts beforehand.
One also needs to modify Line 12 in `codes/scripts/marchenko_pastur/no_parallel.sh` to use a proper environment.

Run the following commands to parallelly train tiny MLPs:
```bash
DEVICE=cuda:0 PARTITION=1 TITLE=marchenko_pastur bash codes/scripts/marchenko_pastur/half_parallel.sh
DEVICE=cuda:1 PARTITION=2 TITLE=marchenko_pastur bash codes/scripts/marchenko_pastur/half_parallel.sh
DEVICE=cuda:2 PARTITION=2_1 TITLE=marchenko_pastur bash codes/scripts/marchenko_pastur/half_parallel.sh
DEVICE=cuda:3 PARTITION=2_2 TITLE=marchenko_pastur bash codes/scripts/marchenko_pastur/half_parallel.sh
```
Results can be found under `runs/marchenko_pastur/`