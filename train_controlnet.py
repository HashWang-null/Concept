import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image

from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from torch.utils.data import DataLoader
from packaging import version
from accelerate import Accelerator
import wandb

from data import load_data
from data.datasets import PairedDataset
from utils.config import parse_yaml
from transformers import CLIPTokenizer, CLIPTextModel


def get_null_text_embeds(pretrained_model, device):
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, local_files_only=True, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, local_files_only=True, subfolder='text_encoder').to(device)
    input_ids = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(input_ids)[0].detach()


class Trainer:
    def __init__(self, config):
        self.accelerator = Accelerator(mixed_precision=config.mixed_precision, log_with="wandb")
        self.config = config
        self.device = self.accelerator.device
        self.weight_dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32

        # Models
        self.scheduler = DDPMScheduler.from_pretrained(config.pretrained_model, local_files_only=True, subfolder='scheduler')
        self.unet = UNet2DConditionModel.from_pretrained(config.pretrained_model, local_files_only=True, subfolder='unet')
        self.vae = AutoencoderKL.from_pretrained(config.pretrained_model, local_files_only=True, subfolder='vae')
        self.controlnet = ControlNetModel.from_unet(self.unet)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae.enable_slicing()
        self.controlnet.train()

        if config.gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()

        self.optimizer = torch.optim.AdamW(
            self.controlnet.parameters(), 
            lr=config.learning_rate, 
            betas=(config.adam_beta1, config.adam_beta2), 
            weight_decay=config.adam_weight_decay, 
            eps=config.adam_epsilon
        )

        train_data = getattr(load_data, config.dataset)(train=True)
        self.train_dataset = PairedDataset(train_data['sources'], train_data['targets'], image_size=512)
        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, 
            batch_size=config.train_batch_size, 
            num_workers=config.num_workers, drop_last=False
        )

        self.lr_scheduler = get_scheduler(config.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=config.max_train_steps, num_cycles=config.lr_num_cycles, power=config.lr_power)

        self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler)

        self.criterion = torch.nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

        self.step = 0
        self.max_train_steps = config.max_train_steps
        self.save_interval = config.save_interval

        self.null_embeds = get_null_text_embeds(config.pretrained_model, self.device).to(dtype=self.weight_dtype)

        if self.accelerator.is_main_process:
            os.makedirs(config.logging_dir, exist_ok=True)
            wandb.init(project=config.project_name, config=dict(vars(config)))

        if config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            out_path = os.path.join(self.config.logging_dir, 'ckpt', f'checkpoint.pth')
            self.accelerator.save({
                'step': self.step,
                'model': self.controlnet.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }, out_path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.step = ckpt['step']
        self.controlnet.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    def encode_images(self, images):
        images = images.to(self.vae.device)
        with torch.autocast("cuda"):
            latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        return latents.detach().to(self.device)

    def decode_images(self, latents):
        latents = latents.to(self.vae.device)
        with torch.autocast("cuda"):
            images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            return torch.clamp((images + 1) / 2, 0, 1)

    def train_timesteps(self, batch_size):
        return torch.randint(self.time_range[0], self.time_range[1], (batch_size,), device=self.device).long()

    def next_batch(self):
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            batch = next(self.train_iter)

        latents = self.encode_images(torch.cat([batch['source'], batch['target']], dim=0))
        sources, targets = latents.chunk(2, dim=0)

        timesteps = torch.randint(0, 1000, size=(targets.shape[0],)).to(self.accelerator.device)
        noise = torch.randn_like(targets)
        sample = self.scheduler.add_noise()
        return {
            "sample": sample.to(dtype=self.weight_dtype),
            "t": timesteps,
            "noise": noise,
            "source": batch['source'].to(self.device, dtype=self.weight_dtype),
            "target": batch['target'].to(self.device, dtype=self.weight_dtype),
            "source_latent": sources,
            "target_latent": targets,
        }

    def train_step(self):
        self.controlnet.train()
        step_loss = 0.
        self.train_iter = iter(self.train_dataloader)

        while self.step < self.max_train_steps:
            with self.accelerator.accumulate(self.controlnet):
                data = self.next_batch()
                cond = self.null_embeds.repeat(data['sample'].shape[0], 1, 1)

                with self.accelerator.autocast():
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        data['sample'], data['t'], encoder_hidden_states=cond, controlnet_cond=data['source'], return_dict=False
                    )
                    model_pred = self.unet(
                        data['sample'], data['t'], encoder_hidden_states=cond,
                        down_block_additional_residuals=[x.to(dtype=self.weight_dtype) for x in down_block_res_samples],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                        return_dict=False
                    )[0]
                    loss = self.criterion(model_pred.float(), data['noise'].float())

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                step_loss = loss.item()

            self.step += 1
            if self.accelerator.is_main_process:
                print(f"[step {self.step}] loss: {step_loss:.6f}")
                wandb.log({"train/loss": step_loss, "train/step": self.step})

            if self.step % self.save_interval == 0:
                self.save_checkpoint()
                if self.accelerator.is_main_process:
                    print("Checkpoint saved")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str)
    args = parser.parse_args()

    config = parse_yaml(args.config)
    config.logging_dir = os.path.join(config.logging_dir, os.path.basename(args.config).split('.')[0])

    trainer = Trainer(config)
    trainer.train_step()


if __name__ == '__main__':
    main()
