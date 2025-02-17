#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import math
import json
import os
import random
import shutil
from pathlib import Path

# from torch_cluster import fps
import copy
from copy import deepcopy
from collections import OrderedDict
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from transformers import AutoModel
from PIL import Image

import open_clip
import diffusers
from diffusers import DDPMScheduler, EulerAncestralDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.vae.vae import get_vae_model
from models.diffusion.transformer_vector import SD3Transformer2DModel
from datasets_diffusion import get_dataset

if is_wandb_available():
    import wandb
else:
    print("please install wandb!")
    exit(1)

os.environ["WANDB_API_KEY"] = "your_wandb_key" #### change to your own wandb key
os.environ["WANDB_MODE"] = "offline"


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float,
                        default=1.0, help="The weight of prior preservation loss.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/code/diffusers",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")

    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--start_ema_percent",
        type=float,
        default=0.1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=None, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true",
                        help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-06,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="v_prediction",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--unet_pretrain_ckpt_dir",
        type=str,
        default=None,
        help=(
            "whether load unet state_dict"
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float,
                        default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--do_classifier_free_guidance",
        action="store_true",
        help="if use classifier free guidance"
    )

    parser.add_argument(
        "--validation_images_dir",
        type=str,
        default=None,
        help=(
            "validation images direction"),
    )
    parser.add_argument(
        "--drop_condition_prob",
        type=float,
        default=0.1,
        help=(
            "drop condition probility"),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def make_validation_steps(checkpointing_steps, min_val_step=300, max_val_step=10000):
    """make validation steps list

    Args:
        checkpointing_steps (int): frequency of checkpointing
        min_val_step (int, optional): minimum val step num. Defaults to 300.
        max_val_step (int, optional): maximum val step num. Defaults to 10000.

    Returns:
        list: validation steps list
    """
    validation_steps = []
    base_checkpointing_step = checkpointing_steps // 100
    for i in range(3):
        validation_steps.append(base_checkpointing_step)
        base_checkpointing_step = base_checkpointing_step * 10

    validation_steps = [max(min_val_step, min(x, max_val_step))
                        for x in validation_steps]
    print(f"validation_steps:{validation_steps}")
    return validation_steps


def get_validation_step(global_step, validation_steps):
    """get validation step based on global step and validation_step_list

    Args:
        global_step (int): current global step
        validation_steps (int): current validation step

    Returns:
        int: _description_
    """
    if global_step <= validation_steps[0] * 10:
        return validation_steps[0]
    elif global_step < validation_steps[0] * 10 + validation_steps[1] * 10:
        return validation_steps[1]
    else:
        return validation_steps[2]


def main():
    args = parse_args()

    config_path = os.path.join(args.output_dir, "train_configs.json")
    with open(config_path, 'r') as fr:
        configs = json.load(fr)
    configs["exp_dir"] = args.output_dir

    print(f"do_classifier_free_guidance: {args.do_classifier_free_guidance}")

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState(
        ).deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = get_vae_model(configs)
        print("vae loading ckpt...")
        vae_state_dict = torch.load(
            configs["vae_config"]["pretrain_path"], map_location='cpu')["state_dict"]
        new_vae_state_dict = {}
        for key, value in vae_state_dict.items():
            new_vae_state_dict[key[12:]] = value
        vae.load_state_dict(new_vae_state_dict, strict=True)

        print("load dino from:", configs["dino_config"]["pretrain_dir"])
        dino_model = AutoModel.from_pretrained(
            configs["dino_config"]["pretrain_dir"], local_files_only=True)
        print("load clip model from:", configs["clip_config"]["pretrain_dir"])
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-bigG-14', cache_dir=configs["clip_config"]["pretrain_dir"])
    del clip_model.transformer
    del clip_model.vocab_size
    del clip_model.token_embedding
    del clip_model.positional_embedding
    del clip_model.ln_final
    del clip_model.text_projection
    del clip_model.attn_mask
    del clip_model.logit_scale

    del vae.decoder
    del vae.transformer

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    diffusion_config = configs["diffusion_config"]
    unet = SD3Transformer2DModel(**diffusion_config)
    unet.train()

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.unet_pretrain_ckpt_dir is not None:
        unet.load_checkpoint(args.unet_pretrain_ckpt_dir, strict=False)

    # Create EMA for the unet.
    if args.use_ema:
        # Create an EMA of the model for use after training
        ema_unet_model = SD3Transformer2DModel(**diffusion_config)
        ema_unet_model.load_state_dict(unet.state_dict())
        ema_unet = EMAModel(ema_unet_model.parameters(), model_config={})
        ema_unet.to("cpu")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            logger.info(
                "enable_xformers_memory_efficient_attention------------")
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema and (global_step / args.max_train_steps) >= args.start_ema_percent:
                    # if args.use_ema:
                    ema_unet.copy_to(ema_unet_model.parameters())
                    ema_unet_model.save_checkpoint(
                        os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_checkpoint(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema and (global_step / args.max_train_steps) >= args.start_ema_percent:
                ema_unet_model.load_checkpoint(
                    os.path.join(input_dir, "unet_ema"))
                ema_unet.__init__(ema_unet_model.parameters(),
                                  model_config=ema_unet_model.config)
                ema_unet.to(accelerator.device)

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                print("accelerator loaded model")
                model.load_checkpoint(os.path.join(input_dir, "unet"))

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # #### TODO: add gradient_checkpointing for dit model
    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    dataset = get_dataset(configs, data_type="train", resample=False)

    logger.info(f"Dataset size: {len(dataset)}")

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    print(f"args.num_train_epochs: {args.num_train_epochs}\n")  # 6
    print(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}\n")  # 61w
    print(f"args.lr_warmup_steps: {args.lr_warmup_steps}\n")  # 1000
    print(f"accelerator.num_processes: {accelerator.num_processes}\n")  # 48
    print(f"args.max_train_steps: {args.max_train_steps}\n")  # 360w
    print(f"args.drop_condition_prob: {args.drop_condition_prob}\n")
    print(f"args.lr_num_cycles: {args.lr_num_cycles}\n")

    if args.lr_warmup_steps is None:
        args.lr_warmup_steps = args.max_train_steps // 10

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles
    )

    print("prepare accelerator...")
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    print("init wandb...")
    if accelerator.is_main_process:
        wandb_config = {
            "batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "validation_epochs": args.validation_epochs,
            "checkpointing_steps": args.checkpointing_steps,
            "lr_warmup_steps": args.lr_warmup_steps,
            "drop_condition_prob": args.drop_condition_prob,
            "output_dir": args.output_dir
        }
        logs_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        wandb.init(
            dir=logs_dir,
            project=os.path.basename(args.output_dir),
            name=os.path.basename(args.output_dir),
            config=wandb_config)

    # # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # # as these weights are only used for inference, keeping weights in full precision is not required.
    vae.eval()
    dino_model.eval()
    clip_model.eval()
    vae.to(accelerator.device, dtype=torch.float16)
    dino_model.to(accelerator.device, dtype=torch.float16)
    clip_model.to(accelerator.device, dtype=torch.float16)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     tracker_config = dict(vars(args))
    #     tracker_config.pop("validation_prompts")
    #     accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset.__len__()}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_gray = Image.fromarray(
        (np.ones((512, 512, 3)) * 127).astype(np.uint8))
    dino_image_gray = dataset.dino_processor(
        image_gray, return_tensors="pt")["pixel_values"]
    clip_image_gray = dataset.clip_processor(
        image_gray, return_tensors="pt")["pixel_values"]

    dino_empty_local_path = configs["dino_config"]["gray_dino_feature_path"]
    clip_empty_global_path = configs["clip_config"]["gray_clip_feature_path"]
    gray_image_local_embedding = torch.load(
        dino_empty_local_path, map_location="cpu")
    gray_image_global_embedding = torch.load(
        clip_empty_global_path, map_location="cpu")
    print(
        f"gray_image_local_embedding shape: {gray_image_local_embedding.shape}")
    print(
        f"gray_image_global_embedding shape: {gray_image_global_embedding.shape}")

    sample_points_num = configs["data_config"]["sample_points_num"]
    vae_encoder_points_num = configs["data_config"]["vae_encoder_points_num"]

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(
            device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(
            accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                image_dino_conds = batch["image_dino_conds"].squeeze(1)
                image_clip_conds = batch["image_clip_conds"].squeeze(1)
                pcd_surface_normal = batch["pcd_surface_normal"]

                bn, vn, cnd, hnd, wnd = image_dino_conds.shape
                image_dino_conds = image_dino_conds.view(bn*vn, cnd, hnd, wnd)

                _, _, cnc, hnc, wnc = image_clip_conds.shape
                image_clip_conds = image_clip_conds.view(bn*vn, cnc, hnc, wnc)

                # #### fps 点云采样
                # pc = pcd_surface_normal[..., :3]
                # bs, N, D = pc.shape
                # flattened = pc.view(bs*N, D)
                # batch = torch.arange(bs).to(pc.device)
                # batch = torch.repeat_interleave(batch, N)
                # pos = flattened
                # ratio =  vae_encoder_points_num / sample_points_num
                # idx = fps(pos, batch, ratio=ratio)
                # pcd_surface_normal = pcd_surface_normal.view(bs*N, -1)[idx].view(bs, -1, pcd_surface_normal.shape[-1])

                # get image_latents
                with torch.no_grad():
                    image_latents_dino = dino_model(
                        pixel_values=image_dino_conds).last_hidden_state
                image_latents_dino = image_latents_dino.to(
                    dtype=weight_dtype).detach().contiguous()

                with torch.no_grad():
                    image_latents_clip = clip_model.encode_image(
                        image_clip_conds.to(dtype=torch.float16))
                image_latents_clip = image_latents_clip.to(
                    dtype=weight_dtype).detach().contiguous()

                gray_image_local_embedding = gray_image_local_embedding.to(
                    device=image_clip_conds.device)
                gray_image_global_embedding = gray_image_global_embedding.to(
                    device=image_clip_conds.device)

                # ##### save gray image feature
                # with torch.no_grad():
                #     image_gray_latents = dino_model(pixel_values=dino_image_gray.to(device=image_clip_conds.device, dtype=torch.float16)).last_hidden_state
                # gray_image_local_embedding = image_gray_latents.to(dtype=weight_dtype).detach().contiguous()
                # torch.save(gray_image_local_embedding.cpu(), configs["dino_config"]["gray_dino_feature_path"])

                # with torch.no_grad():
                #     image_gray_clip_features = clip_model.encode_image(clip_image_gray.to(device=image_clip_conds.device, dtype=torch.float16))
                # gray_image_global_embedding = image_gray_clip_features.to(dtype=weight_dtype).detach().contiguous()
                # torch.save(gray_image_global_embedding.cpu(), configs["clip_config"]["gray_clip_feature_path"])
                # exit(1)
                # #######

                # get vae latents
                _, kl_embed, _ = vae.encode(pcd_surface_normal.to(
                    device=pcd_surface_normal.device, dtype=torch.float16))  # [B, 256, 64]
                latents = kl_embed.to(dtype=weight_dtype)
                latents = torch.repeat_interleave(latents, vn, dim=0)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (
                    u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=latents.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(
                    timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

                # # Get the text embedding for conditioning
                encoder_hidden_states = image_latents_dino
                if args.do_classifier_free_guidance:
                    negative_encoder_hidden_states = torch.cat(
                        [gray_image_local_embedding] * bsz, dim=0)
                    negative_image_latents_pool = torch.cat(
                        [gray_image_global_embedding] * bsz, dim=0)
                    # negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
                    # negative_image_latents_pool = torch.zeros_like(image_latents_pool)
                    drop_idx = np.where(np.random.rand(
                        encoder_hidden_states.shape[0]) < args.drop_condition_prob)
                    encoder_hidden_states[drop_idx] = negative_encoder_hidden_states[drop_idx]
                    image_latents_clip[drop_idx] = negative_image_latents_pool[drop_idx]

                # Predict the noise residual and compute loss
                model_pred = unet(hidden_states=noisy_model_input,
                                  encoder_hidden_states=encoder_hidden_states,
                                  pooled_projections=image_latents_clip,
                                  timestep=timesteps).sample

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = latents
                else:
                    target = noise - latents

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(
                        model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float())
                     ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema and (global_step / args.max_train_steps) >= args.start_ema_percent:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.detach().item(),
                              "lr": lr_scheduler.get_last_lr()[0]})

                if global_step % args.checkpointing_steps == 0 or (global_step >= args.max_train_steps):
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    wandb.finish()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
