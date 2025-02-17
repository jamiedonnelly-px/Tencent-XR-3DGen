"""
 # @ Copyright: Copyright 2022 Tencent Inc
 # @ Author: weizhe
 # @ Create Time: 2024-11-20 11:00:00
 # @ Description: Training BPT using 🤗 Accelerate.
 """

import logging
import math
import os

import numpy as np
import trimesh
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs
import hydra
from tqdm.auto import tqdm
import transformers
from transformers import get_constant_schedule, get_cosine_schedule_with_warmup
from safetensors import safe_open

from dataset import AMDataset
from model.model import MeshTransformer
from model.serializaiton import BPT_deserialize
from model.data_utils import to_mesh
from utils import joint_filter


logger = get_logger(__name__)


@hydra.main(config_path="config", config_name="train-8k-8-16", version_base="1.2")
def main(config):
    os.makedirs(config.output_dir, exist_ok=True)
    
    accelerator_log_kwconfig = {}
    accelerator_log_kwconfig["mixed_precision"] = (
        "no" if config.weight_dtype == "None" else config.weight_dtype
    )

    if config.with_tracking:
        accelerator_log_kwconfig["log_with"] = config.report_to
        accelerator_log_kwconfig["project_dir"] = config.output_dir

    ddp_kwconfig = DistributedDataParallelKwargs(find_unused_parameters=False)
    # print("Debug: find_unused_parameters=True")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwconfig],
        **accelerator_log_kwconfig,
    )

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Wait for all nodes to be ready.
    logger.info("Waiting for everyone...")
    accelerator.wait_for_everyone()

    logger.info("Load dataset...")

    train_dataset = AMDataset(config.train_data)
    eval_dataset = AMDataset(config.val_data)

    logger.info("Load models...")

    # Create AM model

    model = MeshTransformer(**config.model_config)

    # Create data loader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.train_data.batch_size,
        pin_memory=True,
        num_workers=4,
        collate_fn=train_dataset.collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.val_data.batch_size,
        pin_memory=True,
        num_workers=4,
        collate_fn=eval_dataset.collate_fn,
    )

    # Optimizer
    logger.info("Create optimizer...")
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=config.learning_rate, betas=(0.9, 0.95)
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if config.lr_scheduler_type == "constant_without_warmup":
        lr_scheduler = get_constant_schedule(
            optimizer=optimizer,
        )
    elif config.lr_scheduler_type == "cosine_with_warmup":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps * accelerator.num_processes,
            num_training_steps=config.max_train_steps * accelerator.num_processes,
            num_cycles=0.5,  # The number of waves in the cosine schedule
            # (the defaults is 0.5, just decrease from the max value to 0 following a half-cosine).
            last_epoch=-1,
        )

    # Prepare everything with our `accelerator`.
    logger.info("Prepare everything with our accelerator...")
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )
    # Also checkpoint lr_scheduler for correct resuming.
    accelerator.register_for_checkpointing(lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(
        config.max_train_steps / num_update_steps_per_epoch
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = config.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if config.with_tracking:
        accelerator.init_trackers(
            "AM",
            config={
                "wandb": {"name": os.path.basename(config.config_file_name)}
            },
        )

    # Train!
    total_batch_size = (
        config.train_data.batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {config.train_data.batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(config.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(config.output_dir) if f.is_dir()]
            dirs = [
                os.path.join(config.output_dir, f) for f in dirs if f.startswith("step")
            ]
            dirs.sort(key=os.path.getctime)
            # Sorts folders by date modified, most recent checkpoint is the last
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        elif config.resume_from_checkpoint != "":
            checkpoint_path = config.resume_from_checkpoint
            path = os.path.basename(config.resume_from_checkpoint)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * config.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // config.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    elif "load_pretrained" in config.keys():
        if config.load_pretrained is not None:
            # Manually load a pretrained LRM model weights for init.
            # Usually a model pre-trained on smaller scale or different data.
            # This can possibly reduce training time.
            print("##### LOAD PRETRAINED #####")
            print("Loading pretrained AM model from:\n", config.load_pretrained)

            if not config.conditioner_only:
                try:
                    model.module.load(config.load_pretrained)
                except:
                    model.load(config.load_pretrained)
                    
            else:
                pretrained_dict = torch.load(config.load_pretrained)['model']
                pretrained_dict = {k[12:]: v for k, v in pretrained_dict.items() if k.startswith('conditioner')}    
                try:
                    model.module.conditioner.load_state_dict(pretrained_dict)
                except:
                    model.conditioner.load_state_dict(pretrained_dict)
                


    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    for epoch in range(starting_epoch, config.num_train_epochs):
        torch.cuda.empty_cache()
        model.train()
        if config.with_tracking:
            total_loss = 0
        if (
            config.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            print(
                "Skip the first `n` batches in the dataloader when resuming from a checkpoint..."
            )
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
            print("Done")
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model) and accelerator.autocast():
                codes, pc_normal = batch["codes"], batch["pc_normal"]
                loss = model(
                    codes=codes,
                    cache=None,
                    return_loss=True,
                    return_cache=False,
                    append_eos=True,
                    pc=pc_normal,
                    cond_embeds=None,
                )

                loss_items_dict = {"loss": loss}
                # We keep track of the loss at each epoch
                if config.with_tracking:
                    total_loss += loss.detach().float()

                accelerator.backward(loss)

                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # lr_scheduler.step()
                progress_bar.update(1)
                completed_steps += 1
                accelerator.log(
                    {
                        "train_loss": loss.detach().float().item(),
                        "loss": loss_items_dict,
                        "grad_norm": grad_norm.float().item(),
                        "Learning_Rate": lr_scheduler.scheduler.get_lr()[0],
                    },
                    step=completed_steps,
                )
                if accelerator.mixed_precision == "fp16":
                    accelerator.log(
                        {"gradient_scaler_scale": accelerator.scaler.get_scale()},
                        step=completed_steps,
                    )

            if isinstance(checkpointing_steps, int):
                if (
                    (completed_steps > 0)
                    and (completed_steps % checkpointing_steps == 0)
                    and accelerator.is_main_process
                ):
                    output_dir = f"step_{completed_steps}"
                    if config.output_dir is not None:
                        output_dir = os.path.join(config.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= config.max_train_steps:
                break

            if (
                (completed_steps > 0)
                and (completed_steps % config.eval_steps == 0)
                and accelerator.is_main_process
                and accelerator.sync_gradients
            ):
                model.eval()
                eval_metric = None

                for step, batch in tqdm(
                    enumerate(eval_dataloader), total=config.max_eval_samples
                ):
                    with torch.no_grad():
                        # Make inputs
                        codes, pc_normal = batch["codes"], batch["pc_normal"]
                        loss = model(
                            codes=codes,
                            cache=None,
                            return_loss=True,
                            return_cache=False,
                            append_eos=True,
                            pc=pc_normal,
                            cond_embeds=None,
                        )
                        eval_metric = {
                            "loss": loss.data.cpu().numpy(),
                        }

                        try:
                            eval_codes = model.module.generate(
                                batch_size=1,
                                temperature=1.0,
                                pc=pc_normal,
                                filter_logits_fn=joint_filter,
                                filter_kwargs=dict(k=50, p=0.95),
                                return_codes=True,
                            )
                        except:
                            eval_codes = model.generate(
                                batch_size=1,
                                temperature=1.0,
                                pc=pc_normal,
                                filter_logits_fn=joint_filter,
                                filter_kwargs=dict(k=50, p=0.95),
                                return_codes=True,
                            )

                        coords = []
                        try:
                            # decoding codes to coordinates
                            for i in range(len(eval_codes)):
                                code = eval_codes[i]
                                code = code[code != config.model_config.pad_id].cpu().numpy()
                                vertices = BPT_deserialize(
                                    code,
                                    block_size=config.model_config.block_size,
                                    offset_size=config.model_config.offset_size,
                                    use_special_block=config.model_config.use_special_block,
                                )
                                coords.append(vertices)
                        except:
                            coords.append(np.zeros(3, 3))

                        uid = f"eval_{step}"
                        vertices = coords[0]
                        faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
                        mesh = to_mesh(
                            vertices, faces, transpose=False, post_process=True
                        )
                        num_faces = len(mesh.faces)
                        # set the color for mesh
                        face_color = np.array([120, 154, 192, 255], dtype=np.uint8)
                        face_colors = np.tile(face_color, (num_faces, 1))
                        mesh.visual.face_colors = face_colors
                        output_dir = os.path.join(
                            config.output_dir, f"validation_{completed_steps}"
                        )
                        os.makedirs(output_dir, exist_ok=True)
                        mesh.export(f"{output_dir}/{uid}_mesh.obj")

                        pcd = pc_normal.data.cpu().numpy()[0]
                        point_cloud = trimesh.points.PointCloud(pcd[..., 0:3])
                        point_cloud.export(f"{output_dir}/{uid}_pc.ply", "ply")

                    if step > config.max_eval_samples:
                        break

                torch.cuda.empty_cache()

                logger.info(f"epoch {epoch}: {eval_metric}")

                if config.with_tracking:
                    accelerator.log(
                        {
                            "eval_metrics": eval_metric,
                            # "eval_imgs": stacked_images,
                        },
                        step=completed_steps,
                    )
                model.train()

            if (config.checkpointing_steps == "epoch") and accelerator.is_main_process:
                output_dir = f"epoch_{epoch}"
                if config.output_dir is not None:
                    output_dir = os.path.join(config.output_dir, output_dir)
                accelerator.save_state(output_dir)

    if config.with_tracking:
        accelerator.end_training()

    if config.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            config.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )


if __name__ == "__main__":
    main()
