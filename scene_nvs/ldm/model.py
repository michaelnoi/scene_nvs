import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import wandb


class SceneNVSNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="unet"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="text_encoder"
        )
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="scheduler"
        )

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

    def forward(self, image_cond, image_target, T):
        x = self.vae.encode(image_target).latent_dist.sample()
        x = self.vae.config.scaling_factor * x

        noise = torch.randn_like(x).to(self.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (x.shape[0],),
            device=self.device,
        )
        timesteps = timesteps.long()
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

        uncond_input = self.tokenizer(
            [""] * 1, padding="max_length", return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

        condition = T

        # Get the model prediction with conditioning
        pred = self.unet(
            sample=noisy_x,
            timestep=timesteps,
            encoder_hidden_states=uncond_embeddings,
            class_labels=condition,
        ).sample

        return pred, noise, timesteps, x

    def training_step(self, batch, batch_idx):
        image_cond = batch["image_cond"]
        image_target = batch["image_target"]
        T = batch["T"]

        # permute (B, H, W, C) to (B, C, H, W)
        image_cond = image_cond.permute(0, 3, 1, 2)
        image_target = image_target.permute(0, 3, 1, 2)

        pred, noise, timesteps, x = self.forward(image_cond, image_target, T)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        self.log("train_loss", loss)

        uncond_input = self.tokenizer(
            [""] * 1, padding="max_length", return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
        condition = T
        if self.global_step % 5 == 0:
            num_inference_steps = 20  # Number of denoising steps

            generator = torch.manual_seed(0)
            latents = torch.randn(
                (1, self.unet.in_channels, 256 // 8, 256 // 8),
                dtype=torch.float16,
                generator=generator,
            )
            latents = latents.to(self.device)
            self.noise_scheduler.set_timesteps(num_inference_steps)
            for t in tqdm(self.noise_scheduler.timesteps):
                t = t.long()
                latent_model_input = self.noise_scheduler.scale_model_input(latents, t)
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=uncond_embeddings,
                        class_labels=condition,
                    ).sample

                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

            latents = 1 / self.vae.config.scaling_factor * latents
            with torch.no_grad():
                image = self.vae.decode(latents).sample

            grid = torchvision.utils.make_grid(image, nrow=2)
            im = grid.permute(1, 2, 0).clip(-1, 1) * 0.5 + 0.5
            im = im.cpu()
            im = Image.fromarray(np.array(im * 255).astype(np.uint8))
            logger = self.logger.experiment
            logger.log({"Sample generations": wandb.Image(im)})

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        return optimizer
