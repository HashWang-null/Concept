import argparse
from typing import List, Dict, Optional, OrderedDict, Any
import contextlib
import copy
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from copy import deepcopy

import json
import numpy as np
import torch
from torch import nn
import torch.utils.checkpoint

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from packaging import version
from PIL import Image
from torchvision import transforms

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
import transformers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from diffusers.utils import get_logger
from diffusers.optimization import get_scheduler
from diffusers.models.attention import JointTransformerBlock
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid

from data import load_data
from data.dataset import PairedDataset
from utils.config import parse_yaml

logger = get_logger(__name__)


# Copied from dreambooth sd3 example
def load_text_encoders(config, class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision, variant=config.variant
    )
    text_encoder_two = class_two.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=config.revision, variant=config.variant
    )
    text_encoder_three = class_three.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=config.revision, variant=config.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def transformer_block_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    joint_to_q,
    joint_to_k,
    joint_to_v,
    joint_scale = 0.2,
    x_scale = 1.0,
    y_scale = 1.0,
    x_ids = None,
    y_ids = None,
    **kwargs,
):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    if self.context_pre_only:
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
    else:
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

    img_q = self.attn.to_q(norm_hidden_states)
    img_k = self.attn.to_k(norm_hidden_states)
    img_v = self.attn.to_v(norm_hidden_states)
    
    joint_img_q = joint_to_q(norm_hidden_states[x_ids+y_ids])
    joint_img_k = joint_to_k(norm_hidden_states[y_ids+x_ids])
    joint_img_v = joint_to_v(norm_hidden_states[y_ids+x_ids])
    
    inner_dim = img_v.shape[-1]
    head_dim = inner_dim // self.attn.heads
    batch_size = norm_encoder_hidden_states.shape[0]
    
    img_q = img_q.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    img_k = img_k.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    img_v = img_v.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    
    joint_img_q = joint_img_q.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    joint_img_k = joint_img_k.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    joint_img_v = joint_img_v.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    
    txt_q = self.attn.add_q_proj(norm_encoder_hidden_states)
    txt_k = self.attn.add_k_proj(norm_encoder_hidden_states)
    txt_v = self.attn.add_v_proj(norm_encoder_hidden_states)
    
    txt_q = txt_q.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    txt_k = txt_k.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    txt_v = txt_v.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
    
    img_txt_q = torch.cat([img_q, txt_q], dim=2)
    img_txt_k = torch.cat([img_k, txt_k], dim=2)
    img_txt_v = torch.cat([img_v, txt_v], dim=2)
    

    # For original image-txt joint attention
    img_txt_attn = F.scaled_dot_product_attention(
        img_txt_q, img_txt_k, img_txt_v
    )
    img_txt_attn = img_txt_attn.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
    img_attn_out, txt_attn_out = (
        img_txt_attn[:, : hidden_states.shape[1]],
        img_txt_attn[:, hidden_states.shape[1] :],
    )
    
    # For joint Attention Output
    joint_attn = F.scaled_dot_product_attention(
        joint_img_q, joint_img_k, joint_img_v
    ).transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
    
    joint_attn_x, joint_attn_y = joint_attn.chunk(2, dim=0)
    joint_attn_x, joint_attn_y = x_scale * joint_attn_x, y_scale * joint_attn_y
    joint_output = torch.zeros_like(joint_attn)
    
    x_index = torch.Tensor(x_ids).to(dtype=torch.int64, device=hidden_states.device)
    y_index = torch.Tensor(y_ids).to(dtype=torch.int64, device=hidden_states.device)
    x_index = x_index.view(-1, 1, 1).expand(-1, *joint_attn.shape[1:])
    y_index = y_index.view(-1, 1, 1).expand(-1, *joint_attn.shape[1:])
    joint_output = joint_output.scatter_reduce(dim=0, index=x_index, src=joint_attn_x, reduce='sum')
    joint_output = joint_output.scatter_reduce(dim=0, index=y_index, src=joint_attn_y, reduce='sum')
    
    # Add joint attention output BEFORE attn.to_out projection
    img_attn_out = (1.0 - joint_scale) * img_attn_out + joint_scale * joint_output
    
    # to out projection
    img_attn_out = self.attn.to_out[0](img_attn_out)
    if not self.context_pre_only:
        txt_attn_out = self.attn.to_add_out(txt_attn_out)
    
    attn_output, context_attn_output = img_attn_out, txt_attn_out
    
    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
 
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output

    # Process attention outputs for the `encoder_hidden_states`.
    if self.context_pre_only:
        encoder_hidden_states = None
    else:
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

    return encoder_hidden_states, hidden_states


class JointAdapter(nn.Module):
    def __init__(self, transformer:SD3Transformer2DModel):
        super().__init__()
        self.joint_q_list = nn.ModuleList([])
        self.joint_k_list = nn.ModuleList([])
        self.joint_v_list = nn.ModuleList([])
        
        for index_block, block in enumerate(transformer.transformer_blocks):
            self.joint_q_list.append(deepcopy(block.attn.to_q))
            self.joint_k_list.append(deepcopy(block.attn.to_k))
            self.joint_v_list.append(deepcopy(block.attn.to_v))


class AdaptedSD3Transformer(SD3Transformer2DModel):
    def init_joint_adapter(self):
        self.adapter = JointAdapter(self)
        self.use_adapter = True
    
    def set_adapter(self, use_adapter=True):
        self.use_adapter = use_adapter
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        
        if self.use_adapter:
            index_block = 0
            for block, joint_to_q, joint_to_k, joint_to_v in zip(
                self.transformer_blocks,
                self.adapter.joint_q_list,
                self.adapter.joint_k_list,
                self.adapter.joint_v_list,
            ):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_to_q=joint_to_q,
                    joint_to_k=joint_to_k,
                    joint_to_v=joint_to_v,
                    **kwargs,
                )
                index_block += 1
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    **kwargs,
                )
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
        return (output,)
    

def prepare_model(model: SD3Transformer2DModel, train=True):
    model.__class__ = AdaptedSD3Transformer
    JointTransformerBlock.forward = transformer_block_forward
    model.init_joint_adapter()
    if train:
        for param in model.parameters():
            param.requires_grad = False
        trainable_params = dict(model.adapter.named_parameters())
        for name, param in trainable_params.items():
            param.requires_grad = True
            if param.dtype != torch.float32:
                param.data = param.data.float()
                param.requires_grad = True
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable params number: {total_trainable}")
        return trainable_params
    else:
        model.requires_grad_(False)


class Trainer:
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Logging path and accelerator
        logging_dir = Path(config.output_dir, config.logging_dir)
        accelerator_project_config = ProjectConfiguration(
            project_dir=config.output_dir, logging_dir=logging_dir
        )
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with=config.report_to,  # default as wandb
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
        self.accelerator = accelerator

        # Logger
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        if self.accelerator.is_main_process:
            if is_wandb_available() and self.accelerator.tracker is not None:
                import wandb
                wandb.config.update(vars(config))

        # If passed along, set the training seed now.
        if config.seed is not None:
            logger.info(f"Using seed {config.seed}")
            set_seed(config.seed)

        # Handle the logging dir creation
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            logger.info(f"Using config file: {self.config.config_name}")
            shutil.copy(self.config.config_path, logging_dir / f"{self.config.config_name}.yaml")


        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        logger.info(f"Using weight dtype: {self.weight_dtype}")
        
        # Load the tokenizer
        self.tokenizer1 = CLIPTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=config.revision,
        )
        self.tokenizer2 = CLIPTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=config.revision,
        )
        self.tokenizer3 = T5TokenizerFast.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="tokenizer_3",
            revision=config.revision,
        )

        # load components
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            config.pretrained_model_name_or_path, config.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            config.pretrained_model_name_or_path, config.revision, subfolder="text_encoder_2"
        )
        text_encoder_cls_three = import_model_class_from_model_name_or_path(
            config.pretrained_model_name_or_path, config.revision, subfolder="text_encoder_3"
        )
        self.text_encoder1, self.text_encoder2, self.text_encoder3 = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
        )
        self.vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="vae",
            revision=config.revision,
            variant=config.variant,
        )
        self.vae.requires_grad_(False)
        self.text_encoder1.requires_grad_(False)
        self.text_encoder2.requires_grad_(False)
        self.text_encoder3.requires_grad_(False)
        if config.upcast_vae:
            self.vae.to(accelerator.device, dtype=torch.float32)
        else:
            self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder1.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder2.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder3.to(accelerator.device, dtype=weight_dtype)
        logger.info("Fixed components loaded.")

        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Model for train
        transformer = SD3Transformer2DModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="transformer", revision=config.revision, variant=config.variant
        )
        params_to_optimize = prepare_model(transformer, train=True)
        self.named_trainable_params = params_to_optimize
        self.trainable_params = list(params_to_optimize.values())
        
        if config.scale_lr:
            config.learning_rate = (
                config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
            ) 
        logger.info(f"Using learning rate:{config.learning_rate}")
        
        optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=config.max_train_steps * accelerator.num_processes,
            num_cycles=config.lr_num_cycles,
            power=config.lr_power,
        )

        train_dataset, train_dataloader = self.prepare_dataset_and_dataloader()
        self.train_dataset = train_dataset
        # Prepare everything with accelerator.
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
        self.train_data_iterator = iter(self.train_dataloader)
        self.global_step = 0
        logger.info("Loading everything OK.")

    def load_checkpoint(self):
        resume_path = self.config.resume_from_checkpoint
        output_dir = self.config.output_dir
        logger.info(f"Will find checkpoint from {output_dir}")
        if resume_path == "auto":
            checkpoints = os.listdir(output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, d))]
            if len(checkpoints) == 0:
                logger.info("No checkpoints found. Starting training from scratch.")
                self.global_step = 0
                return
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]), reverse=True)
            checkpoint_paths = [os.path.join(output_dir, ckpt) for ckpt in checkpoints]
        elif resume_path:
            checkpoint_paths = [resume_path]
        else:
            self.global_step = 0
            return

        for checkpoint_path in checkpoint_paths:
            try:
                if not os.path.exists(checkpoint_path):
                    continue
                self.accelerator.load_state(checkpoint_path)

                if self.accelerator.is_main_process:
                    state_file = os.path.join(checkpoint_path, "training_state.json")
                    if os.path.exists(state_file):
                        with open(state_file, "r") as f:
                            training_state = json.load(f)
                            global_step = training_state.get("global_step", 0)
                    else:
                        logger.warning(f"{state_file} not found. Global_step set to 0")
                        global_step = 0
                else:
                    global_step = 0

                global_step_tensor = torch.tensor(global_step, device=self.accelerator.device)
                global_step_tensor = self.accelerator.broadcast(global_step_tensor)
                self.global_step = global_step_tensor.item()

                logger.info(f"Successfully resumed from checkpoint {checkpoint_path} at global step {self.global_step}")
                return
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {checkpoint_path}. Error: {str(e)}")
                continue

        logger.warning("All checkpoint loading attempts failed. Starting training from scratch.")
        self.global_step = 0

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
            if self.config.checkpoints_total_limit is not None:
                checkpoints = os.listdir(self.config.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= self.config.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(self.config.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
            self.accelerator.save_state(save_path)
            state_file = os.path.join(save_path, "training_state.json")
            with open(state_file, "w") as f:
                json.dump({"global_step": self.global_step}, f)

            logger.info(f"Saved state to {save_path}")
        self.accelerator.wait_for_everyone()
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def next_batch(self):
        try:
            batch = next(self.train_data_iterator)
        except StopIteration:
            self.train_data_iterator = iter(self.train_dataloader)
            batch = next(self.train_data_iterator)
        
        source = batch["source"].to(self.accelerator.device, dtype=self.weight_dtype)
        target = batch["target"].to(self.accelerator.device, dtype=self.weight_dtype)

        source_latent = self.encode_images(source)
        target_latent = self.encode_images(target)
        source_prompt = batch["source_prompt"]
        target_prompt = batch["target_prompt"]
        source_prompt_embeds, source_pooled_prompt_embeds = self.encode_prompt(source_prompt)
        target_prompt_embeds, target_pooled_prompt_embeds = self.encode_prompt(target_prompt)
        
        return dict(
            source=source,
            target=target,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            source_latent=source_latent,
            target_latent=target_latent,
            source_prompt_embeds=source_prompt_embeds,
            target_prompt_embeds=target_prompt_embeds,
            source_pooled_prompt_embeds=source_pooled_prompt_embeds,
            target_pooled_prompt_embeds=target_pooled_prompt_embeds,
        )

    def train_step(self):
        batch = self.next_batch()
        with self.accelerator.accumulate(self.model):
            bsz = batch["source"].shape[0]
            latents = torch.cat([batch["source_latent"], batch["target_latent"]], dim=0)
            prompt_embeds = torch.cat(
                [batch["source_prompt_embeds"], batch["target_prompt_embeds"]], dim=0
            )
            pooled_prompt_embeds = torch.cat(
                [batch["source_pooled_prompt_embeds"], batch["target_pooled_prompt_embeds"]], dim=0
            )
            
            # Sample timesteps
            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.config.weighting_scheme,
                batch_size=bsz*2,
                logit_mean=self.config.logit_mean,
                logit_std=self.config.logit_std,
                mode_scale=self.config.mode_scale,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)

            # Add noise according to flow matching.
            noise = torch.randn_like(latents)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
            
            # For Joint Attention
            x_ids = list(range(bsz))
            y_ids = list(range(bsz, bsz*2))
            joint_scale = 0.2
            x_scale = 1.0
            y_scale = 1.0

            # Predict the noise residual
            model_pred = self.model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
                x_ids = x_ids,
                y_ids = y_ids,
                joint_scale = joint_scale,
                x_scale = x_scale,
                y_scale = y_scale,
            )[0]

            # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
            # Preconditioning of the model outputs.
            if self.config.precondition_outputs:
                model_pred = model_pred * (-sigmas) + noisy_model_input

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.config.weighting_scheme, sigmas=sigmas)

            # flow matching loss
            if self.config.precondition_outputs:
                target = latents
            else:
                target = noise - latents

            # Compute regular loss.
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.trainable_params, self.config.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)
            
            return loss
        
    def train(self):
        total_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Total optimization steps = {self.config.max_train_steps}")
        logger.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Current global step = {self.global_step}")
        
        while self.global_step < self.config.max_train_steps:
            loss = self.train_step()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                self.global_step += 1

                if self.accelerator.is_main_process:
                    if self.global_step % self.config.checkpointing_steps == 0:
                        self.save_checkpoint()
            logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}

            self.accelerator.log(logs, step=self.global_step)
        self.accelerator.end_training()

    def _encode_prompt_with_t5(
        self,
        text_encoder,
        tokenizer,
        max_sequence_length,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _encode_prompt_with_clip(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
    ):
        text_encoders = [self.text_encoder1, self.text_encoder2, self.text_encoder3]
        tokenizers = [self.tokenizer1, self.tokenizer2, self.tokenizer3]
        
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_tokenizers = tokenizers[:2]
        clip_text_encoders = text_encoders[:2]

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                device=text_encoder.device,
                num_images_per_prompt=1,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = self._encode_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            max_sequence_length=256,
            prompt=prompt,
            num_images_per_prompt=1,
            device=text_encoders[-1].device,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def encode_images(self, pixel_values):
        pixel_values = pixel_values.to(dtype=self.vae.dtype)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.weight_dtype)
        return latents

    @torch.no_grad()
    def decode_images(self, latents):  # [-1, 1]
        latents = latents.to(self.device, dtype=self.weight_dtype)
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        images = self.pipe.vae.decode(latents, return_dict=False)[0]
        images = torch.clip(images, -1.0, 1.0)
        return images
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config path', default="config/dev.yaml", type=str)
    args = parser.parse_args()
    config = parse_yaml(args.config)

    config_name = os.path.basename(args.config).split('.')[0]
    config.output_dir = os.path.join(config.output_dir, config_name)
    config.config_path = args.config
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
