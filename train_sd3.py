import argparse
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

import accelerate
from torch import nn
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.testing_utils import backend_empty_cache
from diffusers.utils.torch_utils import is_compiled_module


logger = get_logger(__name__)


# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three):
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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--num_extra_conditioning_channels",
        type=int,
        default=0,
        help="Number of extra conditioning channels for controlnet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--upcast_vae",
        action="store_true",
        help="Whether or not to upcast vae to fp32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
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
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
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
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
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
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
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
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--dataset_preprocess_batch_size", type=int, default=1000, help="Batch size for preprocessing dataset."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `config.validation_prompt` multiple times: `config.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if config.dataset_name is None and config.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if config.dataset_name is not None and config.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if config.proportion_empty_prompts < 0 or config.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if config.validation_prompt is not None and config.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if config.validation_prompt is None and config.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        config.validation_image is not None
        and config.validation_prompt is not None
        and len(config.validation_image) != 1
        and len(config.validation_prompt) != 1
        and len(config.validation_image) != len(config.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if config.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


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
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
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
    

def prepare_model(model: SD3Transformer2DModel):    
    model.__class__ = AdaptedSD3Transformer
    JointTransformerBlock.forward = transformer_block_forward
    model.init_joint_adapter()


# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
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


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
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


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


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
            log_with=config.report_to,
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

        # If passed along, set the training seed now.
        if config.seed is not None:
            logger.info("Using seed", config.seed)
            set_seed(config.seed)

        # Handle the logging dir creation
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)

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

        # TODO PrepareModel
        transformer = SD3Transformer2DModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="transformer", revision=config.revision, variant=config.variant
        )
        transformer.requires_grad_(False)
        
        if config.scale_lr:
            config.learning_rate = (
                config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
            ) 
        logger.info(f"Using learning rate:{config.learning_rate}")
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )

        # with accelerator.main_process_first():

        train_dataset = make_train_dataset(config, self.tokenizer1, self.tokenizer2, self.tokenizer3, accelerator)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=config.train_batch_size,
            num_workers=config.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
        if config.max_train_steps is None:
            config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=config.max_train_steps * accelerator.num_processes,
            num_cycles=config.lr_num_cycles,
            power=config.lr_power,
        )

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
        if overrode_max_train_steps:
            config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_config = dict(vars(args))

            # tensorboard cannot handle list types for config
            tracker_config.pop("validation_prompt")
            tracker_config.pop("validation_image")

            accelerator.init_trackers(config.tracker_project_name, config=tracker_config)

        

    def train(self):
        total_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Total optimization steps = {self.config.max_train_steps}")
        logger.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if config.resume_from_checkpoint:
            if config.resume_from_checkpoint != "latest":
                path = os.path.basename(config.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(config.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                config.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(config.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = self.noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = self.noise_scheduler.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        image_logs = None
        for epoch in range(first_epoch, config.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(controlnet):
                    # Convert images to latent space
                    pixel_values = batch["pixel_values"].to(dtype=self.vae.dtype)
                    model_input = self.vae.encode(pixel_values).latent_dist.sample()
                    model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=config.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=config.logit_mean,
                        logit_std=config.logit_std,
                        mode_scale=config.mode_scale,
                    )
                    indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
                    timesteps = self.noise_scheduler.timesteps[indices].to(device=model_input.device)

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    # Get the text embedding for conditioning
                    prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)

                    # controlnet(s) inference
                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                    controlnet_image = self.vae.encode(controlnet_image).latent_dist.sample()
                    controlnet_image = controlnet_image * self.vae.config.scaling_factor

                    control_block_res_samples = controlnet(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )[0]
                    control_block_res_samples = [sample.to(dtype=weight_dtype) for sample in control_block_res_samples]

                    # Predict the noise residual
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        block_controlnet_hidden_states=control_block_res_samples,
                        return_dict=False,
                    )[0]

                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    # Preconditioning of the model outputs.
                    if config.precondition_outputs:
                        model_pred = model_pred * (-sigmas) + noisy_model_input

                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)

                    # flow matching loss
                    if config.precondition_outputs:
                        target = model_input
                    else:
                        target = noise - model_input

                    # Compute regular loss.
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = controlnet.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=config.set_grads_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % config.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if config.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(config.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= config.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if config.validation_prompt is not None and global_step % config.validation_steps == 0:
                            image_logs = log_validation(
                                controlnet,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                            )

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= config.max_train_steps:
                    break

        # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            controlnet = unwrap_model(controlnet)
            controlnet.save_pretrained(config.output_dir)

            # Run a final round of validation.
            image_logs = None
            if config.validation_prompt is not None:
                image_logs = log_validation(
                    controlnet=None,
                    args=args,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    step=global_step,
                    is_final_validation=True,
                )

            if config.push_to_hub:
                save_model_card(
                    repo_id,
                    image_logs=image_logs,
                    base_model=config.pretrained_model_name_or_path,
                    repo_folder=config.output_dir,
                )
                upload_folder(
                    repo_id=repo_id,
                    folder_path=config.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        accelerator.end_training()

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def compute_text_embeddings(self, batch, text_encoders, tokenizers):
        with torch.no_grad():
            prompt = batch["prompts"]
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, self.config.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(self.accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.accelerator.device)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}


if __name__ == "__main__":
    args = parse_args()
    main(args)