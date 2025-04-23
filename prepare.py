import os
import PIL
from tqdm.auto import tqdm
from typing import Optional, List, Dict, Callable, Union, Any
from types import SimpleNamespace
from functools import partial
import torch
from copy import deepcopy
import inspect
import math
import random
import einops
import numpy as np
from torch import nn
import shutil
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import JointAttnProcessor2_0
from diffusers import SD3Transformer2DModel
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from diffusers.models.attention import BasicTransformerBlock
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from matplotlib import pyplot as plt
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from torch.utils.tensorboard import SummaryWriter



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


class Runner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        print('Using device:', config.device)
        self.weight_dtype = torch.float16 if config.use_fp16 else torch.float32
        print('Using pretrained Model', config.pretrained_model)
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            config.pretrained_model, local_files_only=True, torch_dtype=self.weight_dtype,
            text_encoder_3=None, tokenizer_3=None,
        ).to(self.device)
        if config.use_t5:
            print("Using T5 text encoder")
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                config.pretrained_model,
                local_files_only=True,
                subfolder='tokenizer_3',
            )
            self.t5_text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
                config.pretrained_model,
                local_files_only=True,
                subfolder='text_encoder_3',
            )
            self.pipe.tokenizer_3 = self.t5_tokenizer
            self.pipe.text_encoder_3 = self.t5_text_encoder
            custom_sequential_cpu_offload(self.t5_text_encoder, self.device)
        
        self.model = self.pipe.transformer
        self.pipe.vae.requires_grad_(False)
        self.pipe.vae.enable_slicing()
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        if self.pipe.text_encoder_3 is not None:
            self.pipe.text_encoder_3.requires_grad_(False)
        print('Pretrained Components loaded.')

        prepare_model(self.model)
        self.model.requires_grad_(False)    
        print('All prepared.')
    
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_embed, pooled_prompt_embed = self.pipe._get_clip_prompt_embeds(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            clip_skip=None,
            clip_model_index=0,
        )
        prompt_2_embed, pooled_prompt_2_embed = self.pipe._get_clip_prompt_embeds(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            clip_skip=None,
            clip_model_index=1,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        t5_prompt_embed = self.pipe._get_t5_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=1,
            max_sequence_length=256,
            device="cpu",
        ).to(clip_prompt_embeds.device, dtype=clip_prompt_embeds.dtype)

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def encode_images(self, images):
        images = images.to(self.device, dtype=self.weight_dtype)
        latents = self.pipe.vae.encode(images).latent_dist.sample()
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def decode_images(self, latents):  # [-1, 1]
        latents = latents.to(self.device, dtype=self.weight_dtype)
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        images = self.pipe.vae.decode(latents, return_dict=False)[0]
        images = torch.clip(images, -1.0, 1.0)
        return images
    
    @torch.no_grad()
    def run_joint(
        self, 
        prompt1, prompt2, neg_prompt1="", neg_prompt2="",
        seed=-1, guidance_scale=7.0, num_inference_steps=28,
        joint_scale=0.2, x_scale=1.0, y_scale=1.0,
    ):
        if seed == -1:
            generator = None
        else:
            generator = torch.Generator(self.device).manual_seed(seed)
        
        latents = torch.randn((2, 16, 128, 128), generator=generator, device=self.device).to(dtype=self.weight_dtype)
        x_ids = [0, 2]
        y_ids = [1, 3]
        
        (
            prompt_embeds1,
            pooled_prompt_embeds1,
        ) = self.get_text_embeds(prompt1)
        (
            negative_prompt_embeds1,
            negative_pooled_prompt_embeds1,
        ) = self.get_text_embeds(neg_prompt1)
        (
            prompt_embeds2,
            pooled_prompt_embeds2,
        ) = self.get_text_embeds(prompt2)
        (
            negative_prompt_embeds2,
            negative_pooled_prompt_embeds2,
        ) = self.get_text_embeds(neg_prompt2)
        
        prompt_embeds = torch.cat([negative_prompt_embeds1, negative_prompt_embeds2, prompt_embeds1, prompt_embeds2], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, pooled_prompt_embeds1, pooled_prompt_embeds2], dim=0)
        
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler,
            num_inference_steps,
            self.device,
            sigmas=None,
        )
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Inference process"):
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=dict(),
                return_dict=False,
                x_ids = x_ids,
                y_ids = y_ids,
                joint_scale = joint_scale,
                x_scale = x_scale,
                y_scale = y_scale,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        images = self.decode_images(latents)
        images = (images+1)/2
        return images

    @torch.no_grad()
    def run_guidance(
        self, 
        prompt1, prompt2, neg_prompt1="", neg_prompt2="", guidance=None,
        seed=-1, guidance_scale=7.0, num_inference_steps=28,
        joint_scale=0.2, x_scale=1.0, y_scale=1.0,
    ):
        if seed == -1:
            generator = None
        else:
            generator = torch.Generator(self.device).manual_seed(seed)
        
        latents = torch.randn((1, 16, 128, 128), generator=generator, device=self.device).to(dtype=self.weight_dtype)
        x_ids = [0, 2]
        y_ids = [1, 3]
        
        (
            prompt_embeds1,
            pooled_prompt_embeds1,
        ) = self.get_text_embeds(prompt1)
        (
            negative_prompt_embeds1,
            negative_pooled_prompt_embeds1,
        ) = self.get_text_embeds(neg_prompt1)
        (
            prompt_embeds2,
            pooled_prompt_embeds2,
        ) = self.get_text_embeds(prompt2)
        (
            negative_prompt_embeds2,
            negative_pooled_prompt_embeds2,
        ) = self.get_text_embeds(neg_prompt2)
        
        prompt_embeds = torch.cat([negative_prompt_embeds1, negative_prompt_embeds2, prompt_embeds1, prompt_embeds2], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, pooled_prompt_embeds1, pooled_prompt_embeds2], dim=0)
        
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler,
            num_inference_steps,
            self.device,
            sigmas=None,
        )
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Inference process"):
            t1 = t.expand(latents.shape[0])
            t2 = torch.Tensor([0]*latents.shape[0]).to(self.device, dtype=torch.int64)
            timestep = torch.cat([t1, t2, t1, t2])
            latent_model_input = torch.cat([latents, guidance, latents, guidance], dim=0)

            noise_pred = self.model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=dict(),
                return_dict=False,
                x_ids = x_ids,
                y_ids = y_ids,
                joint_scale = joint_scale,
                x_scale = x_scale,
                y_scale = y_scale,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            all_latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents, _ = all_latents.chunk(2, dim=0)
        
        images = self.decode_images(latents)
        images = (images+1)/2
        return images


def inference():
    infer_config = SimpleNamespace(
        device = "cuda:0",
        use_fp16 = True,
        pretrained_model = os.path.expanduser("~/.cache/huggingface/hub/stable-diffusion-3-medium"),
        use_t5 = False,
    )
    runner = Runner(infer_config)
    images = runner.run_joint(
        "a cat",
        "a dog", 
        seed=0,
        joint_scale=0.2,
        x_scale=1.0,
        y_scale=1.0,
    )
    save_image(images, "temp4.jpg")


if __name__ == "__main__":
    inference()
