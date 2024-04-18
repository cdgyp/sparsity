#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""
import json
import logging
import math
import os
import sys
import time
import warnings
from tensorboardX import SummaryWriter
from functools import partial
import torch
from dataclasses import asdict, dataclass, field

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from torch import distributed as dist

# import flax
# import jax
# import jax.numpy as jnp
import numpy as np
# import optax
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
# import evaluate
# from flax import jax_utils, traverse_util
# from flax.jax_utils import pad_shard_unpad
# from flax.training import train_state
# from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository, create_repo
from tqdm.auto import tqdm
from torch.distributed import get_rank
from random import shuffle

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BatchEncoding,
    T5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    T5Config,
    is_tensorboard_available,
    set_seed,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    AdamW, Adafactor,
    get_linear_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    enable_full_determinism
)
# from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from transformers.models.t5.modeling_t5 import T5LayerFF, T5DenseActDense
from transformers.utils import send_example_telemetry

import logging
from codes.base import Model, Wrapper

from codes.base.base import BaseModule, Plugin

from  ...base import LossManager, start_tensorboard_server, Model, Wrapper
from ...modules.sparsify import Sparsify
from ...modules.robustness import DoublyBiased
from ...modules.activations import JumpingSquaredReLU, CustomizedReLU, ActivationPosition, CustomizedGELU
from ...modules.hooks import CoefficientPlugin
from .subclasses import CustomSeq2SeqTrainer, CustomSeq2SeqTrainingArgument
from .data import get_eval_dataset


MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments:
    title: str = field()
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training when not logging."}
    )
    gradient_accumulated_steps: int = field(
        default=1
    )
    max_obs_batch_size: int = field(default=1, metadata={"help": "Maximum size of batch size recorded by hooks."})
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    max_steps: float = field(default=100000, metadata={"help": 'Default value is the same as T5 training in "The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers".'})
    max_eval_samples: int = field(default=100000, metadata={"help": "The maximum number of evaluation samples. If the provided evaluation dataset is larger than this number, a random subset will be selected."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    
    resume:str = field(default=None, metadata={"help": "Path to the checkpoint to be resumed"})

    compile: bool = field(default=False, metadata={"help": "Whether to use `torch.compile`. `torch` above 2.0 is required. "})

    throw_empty_samples: bool = field(default=False)

    gradient_checkpointing: bool = field(default=False)


    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

@dataclass
class FinetuningArguments:
    mixed_activation: bool = field(default=False, metadata={"help": "Whether to mix activation functions in order to gradually adapt the replaced activation functions"})
    activation_mixing_steps: int = field(default=3000)
    layernorm_uplifting_steps: int = field(default=3000)
    finetune: str = field(default=None)
    lora_r: int = field(default=192)
    vocab_size: int = field(default=None)

    def __post_init__(self):
        self.activation_mixing_steps = int(self.activation_mixing_steps)
        self.layernorm_uplifting_steps = int(self.layernorm_uplifting_steps)

@dataclass
class EtcArguments:
    scan_eval: bool = field(default=None, metadata={"help": "Evaluate all checkpoints. If this option is turned on, then `resume` indicates the directory holding all checkpoints instead of a single one"})
    dir_to_checkpoints: str = field(default=None)
    
    post_training_only: bool = field(default=False, metadata={"help": ""})
    gradient_density_only: bool = field(default=False, metadata={"help": "If turned on, the whole program is run only to measure the distribution of gradients back propagated to activations. Training, evaluation and most plugins will be turned off. Use in companion with option `--resume`."})
    augmented_flatness_only: bool = field(default=False, metadata={"help": "If turned on, the whole program is run only to measure the distribution of gradients back propagated to activations. Training, evaluation and most plugins will be turned off. Use in companion with option `--resume`."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Sparsification arguments
    db_mlp: bool = field(
        default=False,
    )
    zeroth_bias_clipping : float = field(default=0.1)
    jsrelu: bool = field(
        default=False,
    )
    wide: bool = field(
        default=False,
    )

    restricted_affine: bool = field(
        default=False,
        metadata={
            "help": (
                "Turn off bias parameters in LayerNorm layers and force scaling parameters >= 1"
            )
        }
    )

    magic_synapse: bool = field(
        default=False
    )
    magic_synapse_rho: float = field(
        default=0.1
    )
    

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    from_disk: bool = field(
        default=False, metadata={"help": "The local path to dataset."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )

    def __post_init__(self):
        if self.from_disk is None and self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def shift_tokens_right(input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids

def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        if not isinstance(examples, BatchEncoding):
            batch = BatchEncoding(
                {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
            )
        else:
            batch = examples

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        for key in batch.keys():
            assert isinstance(batch[key], np.ndarray), type(batch[key])
            batch[key] = torch.tensor(batch[key])

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

class HfModel(Model):
    supports_gradient_checkpointing=True
    def __init__(self, main: BaseModule, *plugins: Plugin, writer: SummaryWriter = None, identifier=None):
        super().__init__(main, *plugins, writer=writer, identifier=identifier)
        self.gradient_checkpointing = False
    def gradient_checkpointing_enable(self):
        self.main.gradient_checkpointing_enable()
        for p in self.plugins:
            if hasattr(p, 'gradient_checkpointing_enable'):
                p.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.main.gradient_checkpointing_disable()
        for p in self.plugins:
            if hasattr(p, 'gradient_checkpointing_disable'):
                p.gradient_checkpointing_disable()
    def get_input_embeddings(self):
        return self.main.get_input_embeddings()
class HfWrapper(Wrapper):
    supports_gradient_checkpointing=True
    def __init__(self, model):
        super().__init__(model)
        self.gradient_checkpointing = False
    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()
    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

class T5Sparsify(Sparsify):
    def __init__(self, db_mlp: bool, jsrelu: bool, magic_synapse: bool, restricted_affine: bool = None, zeroth_bias_clipping=0.1, db_mlp_shape=None, rho=0.1, log_per_step=10, scheduling={"activation_mixing_iteration":None, "layernorm_uplifting_iteration": None}, lora_r=None, do_train=True, manual_plugins: 'list[Plugin]'=None,) -> None:
        super().__init__(db_mlp, jsrelu, magic_synapse, restricted_affine, zeroth_bias_clipping, db_mlp_shape, rho, log_per_step, scheduling, lora_r, HfModel, HfWrapper, manual_plugins=manual_plugins)
        self.mlp_types = [T5DenseActDense]
        self.do_train = do_train

    def extract_linear_layers(self, mlp: T5DenseActDense) -> 'dict[str, torch.nn.Linear]':
        return {
            'key': mlp.wi,
            'value': mlp.wo
        }
    
    def is_MLP(self, name: str, module: torch.nn.Module):
        return isinstance(module, T5LayerFF)
    def wrap_MLP(self, path: str, name: str, model: torch.nn.Module, module: T5LayerFF, clipping, shape):
        db_mlp = DoublyBiased(module.DenseReluDense, clipping=clipping, shape=shape, layer_norm=module.layer_norm)
        setattr(module, 'DenseReluDense', db_mlp)
        return model

    def replace_activations(self, model: torch.nn.Module, jsrelu, path='model'):
        if isinstance(jsrelu, str):
            jsrelu = (jsrelu.lower() == 'jsrelu' or jsrelu.lower() == 'jumpring_squared_relu')
        if isinstance(jsrelu, bool):
            if jsrelu:
                make_act = lambda: ActivationPosition(JumpingSquaredReLU())
            else:
                make_act = lambda: ActivationPosition(CustomizedReLU())
        else:
            make_act = jsrelu
            
        for name, module in model.named_children():
            p = '.'.join([path, name])
            if isinstance(module, T5DenseActDense):
                setattr(module, 'act', make_act())
                self.activations.append(p + ': ' + str(module.__class__))
            else:
                self.replace_activations(module, jsrelu=jsrelu, path=p)
        return model

    def magic_synapse_filter(self, name: str, module: torch.nn.Module):
        return '.DenseReluDense.' in name and '.wi' in name

    def load_checkpoint(self, model: torch.nn.Module, path: str, strict: bool):
        path = os.path.join(path, 'pytorch_model.bin')
        checkpoint = torch.load(path, map_location="cpu")
        sd = checkpoint
        if 'main.model.conv_proj.weight' not in model.state_dict():
            sd = {}
            for key, value in checkpoint.items():
                key: str
                if 'conv_proj' in key:
                    key = key.replace('conv_proj', 'conv_proj.conv')
                sd[key] = value
        try:
            matching_status = model.load_state_dict(sd, strict=strict)
        except:
            matching_status = model.main.model.load_state_dict(sd, strict=strict)
        unignorable_unexpected_keys = [
            unexpected_keys 
            for unexpected_keys in matching_status.unexpected_keys 
            if unexpected_keys not in model.main.model._keys_to_ignore_on_load_unexpected
        ]
        assert len(unignorable_unexpected_keys) == 0, (unignorable_unexpected_keys, model.main.model._keys_to_ignore_on_load_unexpected)
        assert all(['lora' in key or key in model.main.model._tied_weights_keys for key in matching_status.missing_keys]), (matching_status.unexpected_keys, [key for key in matching_status.missing_keys if 'lora' not in key and key not in model.main.model._tied_weights_keys])

        return model
    
    def _make_model(self, model, finetuning, has_obs=True, use_mixed_activation=False, force_obs=None):
        from ...modules.hooks import ActivationDistributionPlugin, EffectiveGradientSparsity, SpectralIncreasePlugin, VGradientObservationPlugin
        from ...modules.robustness import RestrictAffinePlugin, ZerothBiasPlugin
        from ...modules.activations import LinearActivationMixing
        obs: 'list[torch.nn.Module]' = [
            RestrictAffinePlugin(
                log_per_step=self.log_per_step, 
                finetuning=(finetuning is not None), 
                uplift_iterations=self.scheduling['layernorm_uplifting_iteration']
            ) if self.restricted_affine else None,
            ActivationDistributionPlugin(self.mlp_types, self.log_per_step),
            ZerothBiasPlugin(self.zeroth_bias_clipping, log_per_step=self.log_per_step) if self.db_mlp else None,
        ] + ([
            SpectralIncreasePlugin(self.mlp_types, self.extract_linear_layers, log_per_step=self.log_per_step),
            EffectiveGradientSparsity(self.mlp_types, self.extract_linear_layers, log_per_step=self.log_per_step),
            VGradientObservationPlugin(mlp_types=self.mlp_types, log_per_step=self.log_per_step),
            LinearActivationMixing(max_iteration=self.scheduling['activation_mixing_iteration']) if use_mixed_activation else None,
        ] if self.do_train else [])
        if not has_obs:
            for ob in obs:
                if ob is not None:
                    assert len(list(ob.parameters())) == 0
            obs = []

        if force_obs is not None:
            if isinstance(force_obs, list):
                obs = force_obs
            elif callable(force_obs):
                obs = force_obs()
            else:
                raise NotImplemented()
        
        model = self.model_type(
            self.wrapper_type(model),
            *obs
        )

        return model

def get_model(config, tokenizer, model_args: ModelArguments, training_args: TrainingArguments, finetuning_args: FinetuningArguments, etc_args: EtcArguments, inputs_length, targets_length):
    
    if model_args.model_name_or_path:
        T5 = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
    else:
        if finetuning_args.finetune is None:
            config.vocab_size = len(tokenizer)
        T5 = T5ForConditionalGeneration(
            config,
        )

    from ...modules.hooks import ActivationHook
    ActivationHook.max_batch_size = training_args.max_obs_batch_size

    from ...modules.hooks import GradientDensityPlugin
    from ...modules.robustness import RestrictAffinePlugin

    if etc_args.post_training_only:
        os.makedirs('runs/T5/' + training_args.title + '/' + ('sparsified' if model_args.jsrelu else 'vanilla'), exist_ok=True)

    if etc_args.post_training_only:
        if etc_args.gradient_density_only: 
            manual_plugins = [
                RestrictAffinePlugin(
                    log_per_step=training_args.logging_steps, 
                    finetuning=False, 
                    uplift_iterations=10000
                ) if model_args.restricted_affine else None,
                GradientDensityPlugin([T5DenseActDense], use_iteration=True, log_range=[-40, 0]),
            ] 
        if etc_args.augmented_flatness_only:
            manual_plugins = [
                RestrictAffinePlugin(
                    log_per_step=training_args.logging_steps, 
                    finetuning=False, 
                    uplift_iterations=10000
                ) if model_args.restricted_affine else None,
                CoefficientPlugin(
                    partial(T5Sparsify.is_MLP, None).
                    partial(T5Sparsify.extract_linear_layers, None),
                    partial(T5Sparsify.activation_function_filter, None)
                )
            ] 

    else:
        manual_plugins = None


    model, writer, output_dir = T5Sparsify(
        db_mlp=model_args.db_mlp,
        jsrelu=model_args.jsrelu,
        magic_synapse=model_args.magic_synapse,
        restricted_affine=model_args.restricted_affine,
        zeroth_bias_clipping=model_args.zeroth_bias_clipping,
        db_mlp_shape=[max(inputs_length, targets_length), T5.config.d_model],
        rho=model_args.magic_synapse_rho,
        log_per_step=training_args.logging_steps,
        scheduling={"activation_mixing_iteration": finetuning_args.activation_mixing_steps, "layernorm_uplifting_iteration": finetuning_args.layernorm_uplifting_steps},
        lora_r=finetuning_args.lora_r,
        do_train=training_args.do_train,
        manual_plugins=manual_plugins
    )(
        'T5', 
        training_args.title,
        T5,
        resume=training_args.resume if not etc_args.post_training_only else 'runs/T5/' + training_args.title + '/' + ("sparsified" if model_args.jsrelu else "vanilla"),
        finetuning=finetuning_args.finetune,
        steps=0,
        start_epoch=0,
        device=int(os.environ['RANK']) if 'RANK' in os.environ else 0,
        tensorboard_server=False,
        mixed_activation=finetuning_args.mixed_activation,
    )

    model.config = T5.config
    training_args.output_dir = output_dir

    if finetuning_args.finetune is not None and training_args.gradient_checkpointing:
        # https://github.com/huggingface/transformers/issues/23170
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if etc_args.post_training_only:
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.requires_grad_(True)
    
    return model, writer
    

    

def get_optimizer_params(model: nn.Module, weight_decay: float):
    """

    by ChatGPT

    Prepares optimizer parameters by excluding LayerNorm's parameters from weight decay.
    
    Args:
    - model (nn.Module): The model for which parameters are prepared.
    - weight_decay (float): The weight decay value.
    
    Returns:
    - List[Union[dict, nn.Parameter]]: List containing parameter groups.
    """

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters

def get_optimizer_scheduler(model: torch.nn.Module, training_args: TrainingArguments):
    param_groups = get_optimizer_params(model, training_args.weight_decay)

    if training_args.adafactor:
        optimizer = Adafactor(param_groups, lr=training_args.learning_rate)
    else:
        optimizer = AdamW(param_groups, lr=training_args.learning_rate, betas=[training_args.adam_beta1, training_args.adam_beta2], eps=training_args.adam_epsilon)
    
    scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=training_args.warmup_steps)
    return optimizer, scheduler

# from torch.utils.tensorboard import SummaryWriter
class TbMetric:
    def __init__(self, eval_steps, loss_manager: LossManager) -> None:
        self.loss_manager = loss_manager
        self.eval_steps = eval_steps
        self.count = 0
    def _compute(self, logits, labels):
        pass
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        return {
            "loss": preds.mean()
        }
    def preprocess_logits_for_metrics(self, eval_preds, labels):
        self.count += 1
        # logits, labels = eval_preds
        logits = eval_preds[1]
        res = self._compute(logits=logits, labels=labels)
        for key, value in res.items():
            self.loss_manager.observe(value, 'eval_metrics', key)
        return res[next(iter(res.keys()))].mean()


class CrossEntropyMetric(TbMetric):
    """
        Directly using the training loss measured on testing samples. 
        Some strange perplexity.
    """
    def __init__(self, eval_steps, loss_manager) -> None:
        super().__init__(eval_steps, loss_manager)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _compute(self, logits, labels):
        labels = labels.to(logits.device)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {
            'ce': loss,
        }


class LoggingCallback(TrainerCallback):
    def __init__(self, resume=False, eval_only=False) -> None:
        super().__init__()
        self.resume = resume
        self.eval_only = eval_only
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        if self.resume:
            model.epoch = state.epoch
            model.iteration = state.global_step
            self.resume = False
        else:
            model.epoch += 1
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        model.iteration += 1
    
    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        model.after_minibatch_backward()
        # losses.observe(model.train_loss(), "loss")
        model.clean()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        model.after_minibatch_backward()
        model.after_backward()

        optimizer = kwargs['optimizer']
        losses: LossManager = model.losses
        losses.observe(optimizer.param_groups[0]["lr"], "lr")
        if model.iteration % args.logging_steps == 0:
            losses.log_losses(model.iteration, True)
            losses.writer.flush()
        losses.reset()
        model.clean()


    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        if self.eval_only:
            model.epoch = state.epoch
            model.iteration = state.global_step

        model.losses.log_losses(model.iteration, testing=True)
        model.losses.writer.flush()
        model.losses.reset()
        model.clean()
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        model.clean()
        


class TokenizingFunction:
    def __init__(self, tokenizer, text_column_name):
        self.tokenizer = tokenizer
        self.text_column_name = text_column_name
    def __call__(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_attention_mask=False)

def myshuffle(x):
    np.random.shuffle(x)
    return x



class SequentialCollator:
    def __init__(self, *collators) -> None:
        self.collators = collators
    def __call__(self, examples) -> Any:
        for c in self.collators:
            examples = c(examples)
        return examples

@dataclass
class TextGroupCollator:
    expanded_inputs_length: int
    pad_token_id: int
    def __call__(self, examples):
        # Concatenate all texts.
        n_samples = len(examples)
        concatenated_examples = {
            k: np.array(list(chain(*[examples[i][k] for i in range(len(examples))]))).flatten() for k in examples[0].keys()
        }
        
        total_length = len(concatenated_examples[next(iter(examples[0].keys()))])
        if total_length < self.expanded_inputs_length * n_samples:
            padded_examples = {
                k: np.pad(v, (0, self.expanded_inputs_length * n_samples - total_length), mode='constant', constant_values=self.pad_token_id)
                for k, v in concatenated_examples.items()
            }
        else:
            padded_examples = {
                k: v[:self.expanded_inputs_length * n_samples]
                for k, v in concatenated_examples.items()
            }

        reshaped_result = {
                k: v.reshape(n_samples, self.expanded_inputs_length) for k, v in padded_examples.items()
            }
        return BatchEncoding(reshaped_result)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, FinetuningArguments, EtcArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, finetuning_args, etc_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args, etc_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments; data_args: DataTrainingArguments; training_args: TrainingArguments; finetuning_args: FinetuningArguments; etc_args: EtcArguments

    if etc_args.post_training_only:
        training_args.do_eval = False
        training_args.learning_rate = 0
        training_args.weight_decay = 0


    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_t5_mlm", model_args, data_args, framework="flax")

    _arg = Seq2SeqTrainingArguments(training_args.output_dir)

    if not etc_args.post_training_only:
        enable_full_determinism(seed=training_args.seed + _arg.local_rank)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")


    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.from_disk:
            try:
                datasets = load_from_disk(
                    data_args.dataset_name,
                )
            except:
                datasets = DatasetDict()
        else:
            datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

        if "validation" not in datasets.keys():
            if data_args.from_disk:
                train_datasets = []
                val_datasets = []
                
                for f in os.listdir(data_args.dataset_name):
                    if os.path.isdir(os.path.join(data_args.dataset_name, f)) and ('train' in f or 'val' in f or 'test' in f):
                        dataset = load_from_disk(
                            os.path.join(data_args.dataset_name, f),
                        )
                        if 'train' in f:
                            train_datasets.append(dataset)
                        else:
                            val_datasets.append(dataset)

                    
                datasets["validation"] = concatenate_datasets(val_datasets).shuffle(seed=42)
                datasets["train"] = concatenate_datasets(train_datasets).shuffle(seed=42)
            else:
                datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                )
                datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.config_name:
        params = {
            "cache_dir": model_args.cache_dir,
            "vocab_size": len(tokenizer),
            "token": model_args.token,
            "torch_dtype": getattr(torch, model_args.dtype),
        }
        config = T5Config.from_pretrained(
            model_args.config_name,
            **params
        )
        if finetuning_args.vocab_size is not None:
            config.vocab_size = finetuning_args.vocab_size
    elif model_args.model_name_or_path:
        config = T5Config.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            torch_dtype=getattr(torch, model_args.dtype),
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type](
            torch_dtype=getattr(torch, model_args.dtype),
        )
        logger.warning("You are instantiating a new config instance from scratch.")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # Since we make sure that all sequences are of the same length, no attention_mask is needed.

    tokenize_function = TokenizingFunction(tokenizer=tokenizer, text_column_name=text_column_name)
    tokenized_datasets = datasets.with_transform(tokenize_function)

    # tokenized_datasets["validation"] = get_eval_dataset(tokenized_datasets["validation"], training_args.max_eval_samples) 




    # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
    # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
    # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )
    print("expanded_inputs_length:", expanded_inputs_length, "targets_length:", targets_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of expanded_inputs_length.

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    model, summary_writer = get_model(config=config, tokenizer=tokenizer, model_args=model_args, training_args=training_args, finetuning_args=finetuning_args, etc_args=etc_args, inputs_length=data_args.max_seq_length, targets_length=targets_length)
    loss_manager = model.losses

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = SequentialCollator(
        TextGroupCollator(
            expanded_inputs_length,
            model.config.pad_token_id
        ),
        DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_probability,
            mean_noise_span_length=data_args.mean_noise_span_length,
            input_length=max_seq_length,
            target_length=targets_length,
            pad_token_id=model.config.pad_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
    )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * torch.cuda.device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * torch.cuda.device_count()

    num_train_steps = len(tokenized_datasets["train"]) // train_batch_size * num_epochs
    num_train_steps = min(num_train_steps, training_args.max_steps)

    # num_of_hosts = jax.process_count()
    # current_host_idx = jax.process_index()

    optimizer_scheduler = get_optimizer_scheduler(model, training_args)

    # Handle the repository creation
    if training_args.push_to_hub:
        # Retrieve of infer repo_name
        repo_name = training_args.hub_model_id
        if repo_name is None:
            repo_name = Path(training_args.output_dir).absolute().name
        # Create repo and retrieve repo_id
        repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
        # Clone repo locally
        repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    trainer_argument = CustomSeq2SeqTrainingArgument(
        os.path.join(training_args.output_dir, 'save'),
        logging_dir=training_args.output_dir,
        overwrite_output_dir=True,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        do_predict=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        logging_steps=training_args.logging_steps,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulated_steps,
        max_obs_batch_size=training_args.max_obs_batch_size,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        **{key: value for key, value in training_args.to_dict().items() if 'adam' in key},
        max_grad_norm=1,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=num_train_steps,
        # lr_scheduler_type='constant',
        # warmup_ratio=0,
        remove_unused_columns=False,
        # reproducibility
        full_determinism=True if not etc_args.post_training_only else False,
        seed=training_args.seed + dist.get_rank(),
        # fp16=True, # open automatic mixed precision
        dataloader_num_workers=16,
        gradient_checkpointing=training_args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=32,
        throw_empty_samples=training_args.throw_empty_samples
        # torch_compile=training_args.compile,
        # torch_compile_mode='reduce-overhead',
    )

    metric = CrossEntropyMetric(training_args.eval_steps, loss_manager=loss_manager)

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=trainer_argument,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=metric.compute_metrics,
        preprocess_logits_for_metrics=metric.preprocess_logits_for_metrics,
        optimizers=optimizer_scheduler,
    )
    
    trainer.add_callback(LoggingCallback(resume=(training_args.resume is not None)))
    logging.getLogger("tensorboard").setLevel(logging.WARNING)

    if not etc_args.scan_eval:
        trainer.train(
            resume_from_checkpoint=training_args.resume
        )
    else:
        if not (not training_args.do_train and training_args.do_eval):
            raise ValueError("Evaluation and no training are assumed under scanning evaluation")
        with torch.no_grad():
            extract_step = lambda x: int(x.split('-')[-1])
            for checkpoint in tqdm(sorted(os.listdir(etc_args.dir_to_checkpoints), key=extract_step)):
                checkpoint_full_path = os.path.join(etc_args.dir_to_checkpoints, checkpoint, 'pytorch_model.bin')
                model.load_state_dict(torch.load(checkpoint_full_path, 'cpu'), strict=True)
                # model.to(int(os.environ['RANK']) if 'RANK' in os.environ else 0)
                model.iteration = extract_step(checkpoint)
                model.eval()
                trainer = CustomSeq2SeqTrainer(
                    model=model,
                    args=trainer_argument,
                    data_collator=data_collator,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    compute_metrics=metric.compute_metrics,
                    preprocess_logits_for_metrics=metric.preprocess_logits_for_metrics,
                    optimizers=optimizer_scheduler,
                )
                trainer.add_callback(LoggingCallback(eval_only=False))
                trainer.evaluate()


if __name__ == "__main__":
    main()
