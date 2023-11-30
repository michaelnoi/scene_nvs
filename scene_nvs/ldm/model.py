import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

import wandb


class SceneNVSNet(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.logger_cfg = cfg.logger
        self.cfg = cfg.model
        self.enable_cfg = self.cfg.guidance.enable
        if self.enable_cfg:
            self.cfg_scale = self.cfg.guidance.cfg_scale
            print("CFG enabled with scale: ", self.cfg_scale)

        # core parts
        self.feature_extractor_vae = CLIPImageProcessor.from_pretrained(
            self.cfg.feature_extractor_vae_path, subfolder="feature_extractor_vae"
        )
        self.vae = AutoencoderKL.from_pretrained(self.cfg.vae.path)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.cfg.unet.path,
            subfolder="unet",
            variant=self.cfg.unet.variant,
        )

        # initialize another fully-connected layer for posed CLIP embedding Appendix C Zero123
        self.pose_projection = torch.nn.Linear(1028, 1024)

        # 2.4 Zero123++ Flex diffuse
        if self.cfg.flex_diffuse.enable:
            L = self.cfg.flex_diffuse.L
            self.flex_diffuse_weights = torch.nn.Parameter(
                torch.zeros(1, L, device=self.device, dtype=torch.float16),
                requires_grad=True,
            )
            with torch.no_grad():
                for i in range(L):
                    self.flex_diffuse_weights[0, i].fill_(i / L)
                print("Flex diffuse weights: ", self.flex_diffuse_weights)

        # CLIP models for conditioning
        self.feature_extractor_clip = CLIPImageProcessor.from_pretrained(
            self.cfg.feature_extractor_clip_path, subfolder="feature_extractor_clip"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.tokenizer.path, subfolder="tokenizer"
        )
        if self.cfg.text_encoder.enable:
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.cfg.text_encoder.path,
                subfolder="text_encoder",
                variant=self.cfg.text_encoder.variant,
            )
            # encode dummy prompt
            self.empty_prompt = self.encode_prompt("")
            print("Empty prompt shape: ", self.empty_prompt.shape)
            # remove text encoder from memory
            # TODO solve this in a better way
            self.text_encoder = None
            torch.cuda.empty_cache()
            self.cfg.text_encoder.enable = False

        if self.cfg.vision_encoder.enable:
            self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.cfg.vision_encoder.path, subfolder="vision_encoder"
            )

        # noise scheduler
        # self.noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        #    "sudo-ai/zero123plus-v1.1", subfolder="scheduler")

        # self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        #    "stabilityai/stable-diffusion-2", subfolder="scheduler")

        # NOTE: EulerDiscreteScheduler supports x0 prediction, but runs into an error at the add_noise step #noqa
        #       EulerAncestralDiscreteScheduler does not support x0 prediction #noqa

        self.noise_scheduler = DDIMScheduler.from_pretrained(
            self.cfg.scheduler.path, subfolder="scheduler"
        )

        self.noise_scheduler.config.prediction_type = (
            self.cfg.scheduler.prediction_overwrite
        )
        self.noise_scheduler.config.beta_schedule = (
            self.cfg.scheduler.beta_schedule_overwrite
        )

        # configure what parts to train
        self.vae.requires_grad_(not self.cfg.vae.freeze)
        self.unet.requires_grad_(not self.cfg.unet.freeze)
        if self.cfg.text_encoder.enable:
            self.text_encoder.requires_grad_(not self.cfg.text_encoder.freeze)

        if self.cfg.vision_encoder.enable:
            self.vision_encoder.requires_grad_(not self.cfg.vision_encoder.freeze)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(prompt, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            encoder_hidden_states_text = self.text_encoder(
                tokens.input_ids  # .to(self.device)
            )[0]

        return encoder_hidden_states_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # TODO: store on disk if memory is an issue
        with torch.no_grad():
            embedding_from_projection = self.vision_encoder(image).image_embeds

        return embedding_from_projection

    def forward(self, image_target: torch.Tensor, encoder_hidden_states: torch.Tensor):
        image_target = image_target.to(self.device).half()

        # 1. Enocde x0 to latent space
        x0 = self.vae.encode(image_target).latent_dist.sample()
        # 2. Scale latent space for unit variance
        x0 = torch.tensor(self.vae.config.scaling_factor, device=self.device) * x0
        # 3. Sample noise and timesteps
        noise = torch.randn_like(x0, device=self.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (x0.shape[0],),
            device=self.device,
        )
        # self.log("sampled_timesteps", timesteps)  # TODO: remove
        # 4. Add noise to x0 according to scheduler and timestep
        noisy_x0 = self.noise_scheduler.add_noise(x0, noise, timesteps)

        if self.enable_cfg:
            # with probability 0.2 set the conditioning to null vector
            if torch.rand(1) < 0.2:
                encoder_hidden_states = torch.zeros(
                    encoder_hidden_states.shape, dtype=torch.float16
                ).to(self.device)

        # Get the model prediction
        unet_output = self.unet(
            sample=noisy_x0,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )

        return unet_output, noise, timesteps, x0

    def sampling_loop(
        self,
        encoder_hidden_states: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        self.unet.eval()

        # generator = torch.manual_seed(0)
        # get latent dims
        num_channels_latents = self.unet.config.in_channels
        height = self.unet.config.sample_size
        width = self.unet.config.sample_size
        # generate random latents
        latents = torch.randn(
            (1, num_channels_latents, height, width),
            dtype=torch.float16,
            generator=None,
        )
        latents = latents.to(self.device)
        # scale latents with initial sigma (often 1? -> check papers)
        latents = self.noise_scheduler.init_noise_sigma * latents

        # set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # create null vector for posed CLIP embedding and input image (3.2 Zero123)
        if self.cfg_scale:
            null_clip = torch.zeros(
                encoder_hidden_states.shape, dtype=torch.float16
            ).to(
                self.device
            )  # ,
            encoder_hidden_states = torch.cat([null_clip, encoder_hidden_states])

        for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
            # CFG
            latents_model_input = (
                torch.cat([latents] * 2) if self.cfg_scale else latents
            )

            # Apply scaling in case scheduler needs it (DDIM does not)
            latents_model_input = self.noise_scheduler.scale_model_input(
                latents_model_input, t
            )
            # t = t.long() needed for some schedulers ?
            with torch.no_grad():
                noise_pred = self.unet(
                    latents_model_input, t, encoder_hidden_states=encoder_hidden_states
                ).sample

            if self.cfg_scale:
                # = 0 -> ingnores conditioning
                # = 1 -> learns vanilla conditioning distribution
                # > 1 -> moves away from from the unconditioned distribution,
                # #i.e. forces the samples to be more conditioned in exchange for less diversity
                (
                    noise_pred_uncond,
                    noise_pred_cond,
                ) = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            scheduler_output = self.noise_scheduler.step(noise_pred, t, latents)
            # Computed sample (x_{t-1}) of previous timestep. prev_sample should be used as next model input in the denoising loop.
            latents = scheduler_output.prev_sample

        return latents

    def latent2img(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Converts a latent vector to an image
        """
        latent = (1 / self.vae.config.scaling_factor) * latent
        with torch.no_grad():
            image = self.vae.decode(latent).sample

        image = torch.clamp(image, -1, 1)
        image = (image + 1) / 2
        image = image.permute(0, 2, 3, 1)
        image = image.detach().cpu()
        image = (image * 255).int()

        return image

    def plot_sampling_loop(
        self,
        pred,
        target,
        timesteps,
        image_target,
        image_cond,
        encoder_hidden_states,
    ) -> None:
        logger = self.logger.experiment
        # Log pred and target images
        pred_decoded = self.latent2img(pred).cpu()
        target_decoded = self.latent2img(target).cpu()
        pred_img = Image.fromarray(pred_decoded[0].numpy().astype(np.uint8))
        target_img = Image.fromarray(target_decoded[0].numpy().astype(np.uint8))

        print("Timesteps: logging ", timesteps)
        logger.log({"Prediction Image": wandb.Image(pred_img)})  # type: ignore
        logger.log({"Target Image": wandb.Image(target_img)})  # type: ignore
        logger.log({"Timestep for prediction": timesteps})

        num_inference_steps = self.cfg.scheduler.num_inference_steps
        latents = self.sampling_loop(encoder_hidden_states, num_inference_steps)
        image = self.latent2img(latents).cpu()

        # Log images
        train_or_val = "train" if self.training else "val"

        im = Image.fromarray(image[0].numpy().astype(np.uint8))

        logger.log({f"Sample generations {train_or_val}": wandb.Image(im)})  # type: ignore

        grid_target = torchvision.utils.make_grid(image_target, nrow=1)
        im_target = grid_target.permute(1, 2, 0).clip(-1, 1) * 0.5 + 0.5
        im_target = im_target.cpu()
        im_target = Image.fromarray(np.array(im_target * 255).astype(np.uint8))
        logger.log({f"Target Image {train_or_val}": wandb.Image(im_target)})  # type: ignore

        grid_cond = torchvision.utils.make_grid(image_cond, nrow=1)
        im_cond = grid_cond.permute(1, 2, 0).clip(-1, 1) * 0.5 + 0.5
        im_cond = im_cond.cpu()
        im_cond = Image.fromarray(np.array(im_cond * 255).astype(np.uint8))
        logger.log({f"Conditioning Image {train_or_val}": wandb.Image(im_cond)})  # type: ignore

    def shared_step(self, batch, batch_idx):
        image_cond = batch["image_cond"].to(self.device)
        image_target = batch["image_target"].to(self.device)
        T = batch["T"]

        # preprocess image
        image_cond = self.feature_extractor_clip(
            images=image_cond, return_tensors="pt"
        ).pixel_values
        image_target = self.feature_extractor_vae(
            images=image_target, return_tensors="pt"
        ).pixel_values

        image_cond = image_cond.to(self.device)
        image_target = image_target.to(self.device)

        image_embeddings = self.encode_image(image_cond)  # shape: [1, 1024]
        # shape: [1, 1, 1024]
        image_embeddings = image_embeddings.unsqueeze(-2)
        T = T.unsqueeze(-2)  # shape: [1,1,4]
        posed_clip_embedding = torch.cat(
            [image_embeddings, T], dim=-1
        )  # shape: [1,1,1028]
        posed_clip_embedding = self.pose_projection(
            posed_clip_embedding
        )  # shape: [1,1,1024]

        posed_clip_embedding = posed_clip_embedding.repeat(
            1, 77, 1
        )  # shape: [1, 77, 1024]

        # 2.4 Zero123++ Flex diffuse
        if self.cfg.flex_diffuse.enable:
            scaled_posed_clip_embedding = (
                self.flex_diffuse_weights.T * posed_clip_embedding
            )
            encoder_hidden_states = (
                self.empty_prompt.to(self.device) + scaled_posed_clip_embedding
            )
        else:
            encoder_hidden_states = posed_clip_embedding

        # print types and shapes
        print("image_cond: ", image_cond.dtype, image_cond.shape)
        print("image_target: ", image_target.dtype, image_target.shape)
        print(
            "encoder_hidden_states: ",
            encoder_hidden_states.dtype,
            encoder_hidden_states.shape,
        )

        return (
            image_target.half(),
            encoder_hidden_states.half(),
            image_cond.half(),
        )  # TODO fix all the conversions

    def training_step(self, batch, batch_idx):
        image_target, encoder_hidden_states, image_cond = self.shared_step(
            batch, batch_idx
        )

        # Notes other inputs
        # added_cond_kwargs: (`dict`, *optional*):
        # A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
        # are passed along to the UNet blocks.
        # class_labels (`torch.Tensor`, *optional*, defaults to `None`):
        # Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.

        unet_output, noise, timesteps, x0 = self.forward(
            image_target, encoder_hidden_states
        )
        pred = unet_output.sample

        # get target of the unet prediction for loss computation
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x0, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = x0
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        self.log("train_loss", loss)  # sync_dist=True)

        if self.global_step % self.logger_cfg.log_image_every_n_steps == 0:
            self.plot_sampling_loop(
                pred,
                target,
                timesteps,
                image_target,
                image_cond,
                encoder_hidden_states,
            )

        return loss

    # TODO: change back again, only for overfitting debug here
    def validation_step(self, batch, batch_idx):
        return 0.0  # dummy
        image_target, encoder_hidden_states, image_cond = self.shared_step(
            batch, batch_idx
        )

        unet_output, noise, timesteps, x0 = self.forward(
            image_target, encoder_hidden_states
        )
        pred = unet_output.sample

        # get target of the unet prediction for loss computation
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x0, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = x0
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        self.log("val_loss", loss)  # sync_dist=True)
        if self.global_step % self.logger_cfg.log_image_every_n_steps == 0:
            self.plot_sampling_loop(
                pred,
                target,
                timesteps,
                image_target,
                image_cond,
                encoder_hidden_states,
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        return optimizer
