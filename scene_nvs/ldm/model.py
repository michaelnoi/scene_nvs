import os
from typing import Optional, Tuple, Union

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# from deepspeed.ops.adam import DeepSpeedCPUAdam
from diffusers import AutoencoderKL, DDIMScheduler, PNDMScheduler, UNet2DConditionModel
from einops import rearrange
from evaluation.metrics import calculate_psnr
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from PIL import Image

# from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from utils.distributed import rank_zero_print

import wandb
from scene_nvs.utils.timings import rank_zero_print_log_time

from .encodings import NeRFEncoding

# from torch_ort.optim import FP16_Optimizer
# mypy: ignore-errors


class DreamPoseAdapter(nn.Module):
    # https://github.com/johannakarras/DreamPose/blob/main/models/unet_dual_encoder.py#L37
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.vae2clip = nn.Linear(1024, 1024)
        self.linear = nn.Linear(81, 77)
        with torch.no_grad():
            # Zero init influence of vae embedding
            self.linear.weight = nn.Parameter(torch.eye(77, 81))

    def forward(self, clip_embedding, vae_embedding):
        # clip_embedding shape: [1, 77, 1024]
        # vae_embedding shape: [1, 4, 64, 64]

        rank_zero_print("clip_embedding shape: ", clip_embedding.shape)
        rank_zero_print("vae_embedding shape: ", vae_embedding.shape)

        vae_embedding = self.pool(vae_embedding)  # shape [1, 4, 32, 32]
        # flatten dim 2,3 of vae_embedding
        vae_embedding = rearrange(
            vae_embedding, "b c h w -> b c (h w)"
        )  # shape [1, 4, 1024]
        vae_embedding = self.vae2clip(vae_embedding)  # shape [1, 4, 1024]

        rank_zero_print("vae_embedding shape: ", vae_embedding.shape)

        # concatenate clip_embedding and vae_embedding
        concat_embedding = torch.cat(
            [clip_embedding, vae_embedding], dim=1
        )  # shape [1, 81, 1024]

        rank_zero_print("CONCAT EMBEDDING SHAPE: ", concat_embedding.shape)

        concat_embedding = rearrange(
            concat_embedding, "b c d -> b d c"
        )  # shape [1, 1024, 81]

        rank_zero_print("CONCAT EMBEDDING SHAPE: ", concat_embedding.shape)

        concat_embedding = self.linear(concat_embedding)  # shape [1, 1024, 77]
        concat_embedding = rearrange(
            concat_embedding, "b d c -> b c d"
        )  # shape [1, 77, 1024]

        return concat_embedding


class DepthFeatureExtractor(nn.Module):
    def __init__(self):
        super(DepthFeatureExtractor, self).__init__()
        # First convolutional layer with 16 filters, filter size 2, stride 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=2)
        # Second convolutional layer with 32 filters, filter size 2, stride 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        # 1x1 convolutional layer to reduce channels from 32 to 1, zero-initialized
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        # Initialize the weights of conv3 to zero
        nn.init.zeros_(self.conv3.weight)
        # Initialize the bias of conv3 to 0
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        # Applying first convolutional layer with ReLU activation
        x = F.relu(self.conv1(x))
        # Applying second convolutional layer with ReLU activation
        x = F.relu(self.conv2(x))
        # Applying the 1x1 convolutional layer
        x = self.conv3(x)
        return x


class LinearFlexDiffuse(torch.nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        self.weights = torch.nn.Parameter(
            torch.zeros(1, L, dtype=torch.float16),
            requires_grad=True,
        )
        with torch.no_grad():
            for i in range(L):
                self.weights[0, i].fill_((i + 1) / L)
            # rank_zero_print("Flex diffuse weights: ", self.weights)

    def forward(self, x):
        scaled_x = self.weights.T * x
        return scaled_x


# from https://github.com/huggingface/diffusers/blob/f72b28c75b2b4b720a5d8de78556694cf4b893fd/src/diffusers/training_utils.py#L30 # noqa


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849 # noqa
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026 # noqa
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


class SceneNVSNet(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.logger_cfg = cfg.logger
        self.data_cfg = cfg.datamodule
        self.image_size = cfg.datamodule.image_size
        self.optimizer_dict = cfg.trainer.optimizer
        self.image_embeds_to_disk = cfg.image_embeds_to_disk
        self.cfg = cfg.model
        self.enable_cfg = self.cfg.guidance.enable
        if self.enable_cfg:
            self.cfg_scale = self.cfg.guidance.cfg_scale
            rank_zero_print("CFG enabled with scale: ", self.cfg_scale)
        if self.cfg.snr_gamma != 0:
            rank_zero_print("SNR gamma enabled: ", self.cfg.snr_gamma)

        # DEPTH CONDITIONING##
        # if self.cfg.depth_conditioning.enable:
        #    rank_zero_print("Depth conditioning enabled")
        #    self.depth_feature_extractor = DepthFeatureExtractor()

        # metrics
        # It is highly recommended to re-initialize the metric per mode (train/test/val)https://lightning.ai/docs/torchmetrics/stable/pages/overview.html
        # we recommend logging the metric object to make sure that metrics are correctly computed and reset
        self.lpips_loss_train = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze", reduction="mean"
        ).requires_grad_(False)
        self.ssim_loss_train = StructuralSimilarityIndexMeasure(
            data_range=(-1, 1), reduction="elementwise_mean"
        ).requires_grad_(False)
        # self.psnr_loss_train = PeakSignalNoiseRatio(
        #    data_range=(-1, 1), reduction="elementwise_mean")
        # Val
        self.lpips_loss_val = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze", reduction="mean"
        ).requires_grad_(False)
        self.ssim_loss_val = StructuralSimilarityIndexMeasure(
            data_range=(-1, 1),
            reduction="elementwise_mean",
        ).requires_grad_(False)
        # self.psnr_loss_val = PeakSignalNoiseRatio(
        #    data_range=(-1, 1), reduction="elementwise_mean")
        # self.fid = FrechetInceptionDistance(feature=64, normalize=False).requires_grad_(
        #     False
        # )

        # core parts
        if "sd-vae-ft-mse" in self.cfg.vae.path:
            self.vae = AutoencoderKL.from_pretrained(self.cfg.vae.path)
        else:
            self.vae = AutoencoderKL.from_pretrained(
                self.cfg.vae.path, subfolder="vae", variant=self.cfg.vae.variant
            )

        in_channels = 4
        if self.cfg.rgb_conditioning.enable:
            rank_zero_print("RGB conditioning enabled")
            in_channels += 3
        else:
            rank_zero_print("RGB conditioning disabled")

        if self.cfg.depth_conditioning.enable and "depth" not in self.cfg.unet.path:
            rank_zero_print("Depth conditioning enabled but not using depth unet")
            in_channels += 1
        elif self.cfg.depth_conditioning.enable and "depth" in self.cfg.unet.path:
            raise NotImplementedError("Not supported anymore")
        else:
            rank_zero_print("Depth conditioning disabled")

        if in_channels > 4:
            rank_zero_print("Using {} channels for conditioning".format(in_channels))
            self.unet = UNet2DConditionModel.from_pretrained(
                self.cfg.unet.path,
                subfolder="unet",
                variant=self.cfg.unet.variant,
                in_channels=in_channels,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
            # load this from a file
            if self.cfg.unet.path == "stabilityai/stable-diffusion-2-1":
                conv_in_weights = torch.load(
                    "/home/checkpoints/conv_in_weights_4_21.pt"
                )
            elif self.cfg.unet.path == "stabilityai/stable-diffusion-2":
                conv_in_weights = torch.load(
                    "/home/checkpoints/conv_in_weights_4.pt"
                )  # shape[320,4,3,3]
            else:
                raise NotImplementedError(  # noqa
                    "Need to load conv_in weights for this model"
                )

            # set first 4 channels of conv_in to the weights
            self.unet.conv_in.weight.data[:, :4, :, :] = conv_in_weights.data
            # zero initialize the last channel
            self.unet.conv_in.weight.data[:, 4, :, :] = 0
            rank_zero_print("Zero initialized last channel of conv_in")
        else:
            rank_zero_print("Using standard unet")
            self.unet = UNet2DConditionModel.from_pretrained(
                self.cfg.unet.path,
                subfolder="unet",
                variant=self.cfg.unet.variant,
            )

        if self.cfg.lora.enable:
            rank_zero_print("LoRA enabled")
            # https://github.com/huggingface/peft/blob/main/examples/lora_dreambooth/train_dreambooth.py#L50
            lora_config = LoraConfig(
                r=self.cfg.lora.rank,
                init_lora_weights=self.cfg.lora.init_weights,
                lora_alpha=self.cfg.lora.alpha,
                target_modules=self.cfg.lora.target_modules,
                # save conv_in weights to be trained without lroa
                modules_to_save=self.cfg.lora.modules_to_save,
            )
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()
            # rank_zero_print(self.unet)

        in_dim = 7  # dimension of rotation and translation
        num_frequencies = 10  # used in NeRF paper
        self.positional_encoding = NeRFEncoding(
            in_dim=in_dim,
            num_frequencies=num_frequencies,
            min_freq_exp=1,
            max_freq_exp=5,
        )

        positional_encoding_shape = 2 * in_dim * num_frequencies
        # initialize another fully-connected layer for posed CLIP embedding Appendix C Zero123
        in_features = 1024 + positional_encoding_shape
        self.pose_projection = torch.nn.Linear(in_features, 1024)

        # print dim of pose projection
        rank_zero_print("Pose projection shape: ", self.pose_projection)
        self.pose_projection.requires_grad_(True)

        # 2.4 Zero123++ Flex diffuse
        if self.cfg.flex_diffuse.enable:
            rank_zero_print("Flex diffuse enabled")
            self.linear_flex_diffuse = LinearFlexDiffuse(self.cfg.flex_diffuse.L)
        # CLIP models for conditioning
        self.feature_extractor_clip = CLIPImageProcessor.from_pretrained(
            self.cfg.feature_extractor_clip_path, subfolder="feature_extractor"
        )
        if self.cfg.vision_encoder.enable:
            self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.cfg.vision_encoder.path, subfolder="image_encoder"
            )
            self.vision_encoder.requires_grad_(not self.cfg.vision_encoder.freeze)

        # DreamPose
        if self.cfg.dreampose_adapter.enable:
            self.dreampose_adapter = DreamPoseAdapter()
            rank_zero_print("DreamPose adapter enabled")

        # CLIP text encoder for encoding the empty prompt or used for conditioning
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.tokenizer.path, subfolder="tokenizer"
        )
        if (
            not os.path.exists(self.cfg.empty_prompt_embed_path)
            or self.cfg.text_encoder.enable
        ):
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.cfg.text_encoder.path,
                subfolder="text_encoder",
                variant=self.cfg.text_encoder.variant,
            )
            self.text_encoder.requires_grad_(not self.cfg.text_encoder.freeze)

            if not os.path.exists(self.cfg.empty_prompt_embed_path):
                self.empty_prompt = self.encode_prompt("")
                empty_prompt_dir = os.path.dirname(self.cfg.empty_prompt_embed_path)
                os.makedirs(empty_prompt_dir, exist_ok=True)
                torch.save(self.empty_prompt, self.cfg.empty_prompt_embed_path)
                if not self.cfg.text_encoder.enable:
                    self.text_encoder = None
                    torch.cuda.empty_cache()

        # get empty prompt embedding
        self.empty_prompt = torch.load(
            self.cfg.empty_prompt_embed_path, map_location=self.device
        )
        # rank_zero_print(
        #     "Empty prompt shape: ", self.empty_prompt.shape
        # )  # shape: [1, 77, 1024]
        # rank_zero_print("Empty prompt requires grad: ", self.empty_prompt.requires_grad)
        self.empty_prompt.requires_grad_(False)  # is false already

        # noise scheduler !!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: NEED TO BE DIFFERENT CLASS FOR EACH MODEL
        if self.cfg.scheduler.type == "DDIM":
            self.noise_scheduler = DDIMScheduler.from_pretrained(
                self.cfg.scheduler.path, subfolder="scheduler"
            )
        elif self.cfg.scheduler.type == "PNDM":
            self.noise_scheduler = PNDMScheduler.from_pretrained(
                self.cfg.scheduler.path, subfolder="scheduler"
            )
        else:
            raise NotImplementedError("Check which scheduler to use")
        self.noise_scheduler.config.prediction_type = (
            self.cfg.scheduler.prediction_overwrite
        )
        self.noise_scheduler.config.beta_schedule = (
            self.cfg.scheduler.beta_schedule_overwrite
        )

        # configure what parts to train
        self.vae.requires_grad_(not self.cfg.vae.freeze)
        # self.unet.requires_grad_(not self.cfg.unet.freeze)

        self.training_step_outputs: list = []
        self.validation_step_outputs: list = []
        self.train_iteration = 0
        self.val_iteration = 0

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(prompt, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            encoder_hidden_states_text = self.text_encoder(
                tokens.input_ids  # .to(self.device)
            )[0]

        return encoder_hidden_states_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding_from_projection = self.vision_encoder(image).image_embeds

        return embedding_from_projection

    def encode_depth(self, depth_map: torch.Tensor) -> torch.Tensor:
        # depth_map B x 1 x H x W

        depth_map = torch.nn.functional.interpolate(
            depth_map,
            size=(self.image_size // 8, self.image_size // 8),
            mode="bicubic",
            align_corners=False,
        )

        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0

        # depth_map = self.depth_feature_extractor(depth_map)

        # B x 1 x 64 x 64
        return depth_map

    def del_vision_encoder(self):
        self.vision_encoder = None
        torch.cuda.empty_cache()

    @rank_zero_only
    @torch.no_grad()
    def save_all_image_embeddings(self, datamodule: pl.LightningDataModule):
        """
        Save all embeddings for the dataset to disk to be able to delete the image encoder.
        This only works for batch size 1 as of now.
        TODO: make this work for batch size > 1
        """

        # ensure self.device is cuda

        self.vision_encoder.eval()
        self.vision_encoder.to("cuda:0")

        # save embeddings for train
        counter = 0
        for loader in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
            rank_zero_print("Checking embeddings for loader: ", loader)
            for batch in tqdm(loader):
                image_cond = batch["image_cond"].to("cuda:0")
                path_cond = batch["path_cond"]
                b = image_cond.size(0)

                for i, path_cond in enumerate(path_cond):
                    directory, filename = os.path.split(path_cond)
                    filename = os.path.splitext(filename)[0]
                    parent_dir = os.path.dirname(directory)
                    directory = os.path.join(parent_dir, "rgb_image_embeddings")
                    os.makedirs(directory, exist_ok=True)
                    save_path = os.path.join(directory, f"{filename}.pt")
                    if not os.path.exists(save_path):
                        # preprocess image
                        counter += 1
                        image_cond_to_encode = self.feature_extractor_clip(
                            images=image_cond.to("cuda:0"), return_tensors="pt"
                        ).pixel_values

                        # get encoder_hidden_states from CLIP vision model concat pose into projection
                        image_embeddings = self.encode_image(
                            image_cond_to_encode.to("cuda:0")
                        )
                        image_embeddings = image_embeddings.unsqueeze(-2)
                        assert image_embeddings.shape == torch.Size([b, 1, 1024])

                        # save embeddings
                        # shape [1, 1024]
                        torch.save(image_embeddings[i], save_path)

        rank_zero_print("Saved not yet saved image embeddings to disk : ", counter)

    def forward(
        self,
        image_target: torch.Tensor,
        posed_clip_embedding: torch.Tensor,
        vae_embedding: torch.Tensor,
        depth_map: torch.Tensor = None,
        rgb_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        image_target = image_target  # .to(self.device).half()

        # 1. Encode x0 to latent space
        x0 = self.vae.encode(image_target).latent_dist.sample()
        # rank_zero_print("x0 shape (vae output): ", x0.shape)
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
        # 4. Add noise to x0 according to scheduler and timestep
        noisy_x0 = self.noise_scheduler.add_noise(x0, noise, timesteps)
        batch_size = posed_clip_embedding.size(0)

        # rank_zero_print("vae_embedding shape: ", vae_embedding.shape)
        # Global Conditioning
        if self.enable_cfg and self.training:
            if torch.rand(1) < 0.05:
                # set clip embedding to 0
                posed_clip_embedding = self.empty_prompt.expand(batch_size, -1, -1)
            if torch.rand(1) < 0.05:
                # set vae embedding to 0
                vae_embedding = torch.zeros_like(vae_embedding, device=self.device)
            if torch.rand(1) < 0.05:
                # set all global conditioning to 0
                posed_clip_embedding = self.empty_prompt.expand(batch_size, -1, -1)
                vae_embedding = torch.zeros_like(vae_embedding, device=self.device)

        # rank_zero_print("posed_clip_embedding shape: ", posed_clip_embedding.shape)
        # rank_zero_print("vae_embedding shape: ", vae_embedding.shape)

        if self.cfg.dreampose_adapter.enable:
            encoder_hidden_states = self.dreampose_adapter(
                posed_clip_embedding, vae_embedding
            )
        else:
            encoder_hidden_states = posed_clip_embedding

        # Local Conditioning
        if self.cfg.depth_conditioning.enable:
            if self.training and torch.rand(1) < 0.05:
                # set depth_map to 0
                depth_map = torch.zeros_like(depth_map, device=self.device)
            noisy_x0 = torch.cat([noisy_x0, depth_map], dim=1)

        if self.cfg.rgb_conditioning.enable:
            if self.training and torch.rand(1) < 0.05:
                # set rgb_cond to 0
                rgb_cond = torch.zeros_like(rgb_cond, device=self.device)
            noisy_x0 = torch.cat([noisy_x0, rgb_cond], dim=1)

        unet_output = self.unet(
            sample=noisy_x0,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )

        # self.log("timesteps", timesteps.float()) not working for multi batch like this

        return unet_output, noise, timesteps, x0

    def sampling_loop(
        self,
        posed_clip_embedding: torch.Tensor,
        vae_embedding: torch.Tensor,
        num_inference_steps: int,
        depth_map: Optional[torch.Tensor] = None,
        rgb_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.unet.eval()

        # generator = torch.manual_seed(0)

        num_channels_latents = 4
        height = (
            self.image_size // 8
        )  # self.unet.config.sample_size #64 if 512, 96 if 768
        width = (
            self.image_size // 8
        )  # self.unet.config.sample_size #64 if 512, 96 if 768
        # generate random latents
        # n_samples = 1 if not self.logger_cfg.n_samples else self.logger_cfg.n_samples
        # encoder_hidden_states = encoder_hidden_states.expand(n_samples, -1, -1)
        if self.cfg.dreampose_adapter.enable:
            encoder_hidden_states = self.dreampose_adapter(
                posed_clip_embedding, vae_embedding
            )
        else:
            encoder_hidden_states = posed_clip_embedding

        n_samples = encoder_hidden_states.size(0)

        latents = torch.randn(
            (n_samples, num_channels_latents, height, width),
            dtype=torch.float16,
            generator=None,
            device=self.device,
        )
        # rank_zero_print("SAMPLING LOOP: latent shape: ", latents.shape)
        latents = latents  # .to(self.device)
        # scale latents with initial sigma (often 1? -> check papers)
        latents = self.noise_scheduler.init_noise_sigma * latents

        # set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # double batch size for CFG with encoder_hidden_states:
        # [empty prompt, posed CLIP embedding]
        if self.cfg_scale:
            encoder_hidden_states = torch.cat(
                [
                    self.empty_prompt.expand(encoder_hidden_states.size(0), -1, -1),
                    encoder_hidden_states,
                ]
            )
            encoder_hidden_states = encoder_hidden_states.half()  # why again ?

            if self.cfg.depth_conditioning.enable:
                depth_map = torch.cat(
                    [torch.zeros_like(depth_map, device=self.device), depth_map]
                )
            if self.cfg.rgb_conditioning.enable:
                rgb_cond = torch.cat(
                    [torch.zeros_like(rgb_cond, device=self.device), rgb_cond]
                )

        # plotting_timesteps = []
        # plotting_latents = []
        # plotting_pred_original_sample = []
        for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
            # CFG
            latents_model_input = (
                torch.cat([latents] * 2) if self.cfg_scale else latents
            )

            if self.cfg.depth_conditioning.enable:
                latents_model_input = torch.cat([latents_model_input, depth_map], dim=1)

            if self.cfg.rgb_conditioning.enable:
                latents_model_input = torch.cat([latents_model_input, rgb_cond], dim=1)

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

            # plot the latent every 10 steps
            # if i % 10 == 0 or i == 0 or i == len(self.noise_scheduler.timesteps) - 1:
            #    plotting_timesteps.append(t)
            #    plotting_latents.append(latents)
            #    plotting_pred_original_sample.append(
            #        scheduler_output.pred_original_sample
            #    )
        # self.plot_timestep(
        #     plotting_latents, plotting_pred_original_sample, plotting_timesteps
        # )
        self.unet.train()
        return latents

    def latent2img(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image
        """
        latent = (1 / self.vae.config.scaling_factor) * latent
        with torch.no_grad():
            image = self.vae.decode(latent).sample

        return image

    def transform_decoded(
        self,
        image: torch.Tensor,
        to_255: Optional[bool] = True,
        return_PIL: Optional[bool] = True,
        return_batch: Optional[bool] = False,
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Convert first image from tensor to PIL image or tensor for visualization
        """
        image = torch.clamp(image, -1, 1)
        image = (image + 1) / 2
        image = image.permute(0, 2, 3, 1)
        image = image.detach().cpu()
        if to_255:
            image = (image * 255).int()
        else:
            image = image.to(torch.float32)

        if return_PIL:
            if return_batch:
                return [
                    Image.fromarray(image.numpy().astype(np.uint8)) for image in image
                ]
            else:
                return Image.fromarray(image[0].numpy().astype(np.uint8))
        else:
            if return_batch:
                return image
            else:
                return image[0]

    def plot_timestep(self, latents, pred_original_sample, timesteps):
        train_or_val = "Train" if self.training else "Val"
        nrows = len(timesteps)
        fig, axs = plt.subplots(nrows, 2, figsize=(12, 5 * nrows))

        for i in range(nrows):
            latents_i = latents[i]
            pred_original_sample_i = pred_original_sample[i]
            decoded_latents = self.transform_decoded(
                self.latent2img(latents_i), return_PIL=False, return_batch=True
            )
            grid = torchvision.utils.make_grid(decoded_latents, nrow=1)
            axs[i, 0].imshow(grid.cpu())
            axs[i, 0].set_title(f"Current x (step {timesteps[i]})")

            pred_x0 = pred_original_sample_i
            pred_x0 = self.transform_decoded(
                self.latent2img(pred_x0), return_PIL=False, return_batch=True
            )
            grid = torchvision.utils.make_grid(pred_x0, nrow=1)
            axs[i, 1].imshow(grid.cpu())
            axs[i, 1].set_title(f"Predicted denoised images (step {timesteps[i]})")

        self.logger.experiment.log(
            {f"{train_or_val}/Sampling Loop": wandb.Image(fig)}
        )  # type: ignore

    def get_sampled_images(
        self,
        posed_clip_embedding: torch.Tensor,
        vae_embedding: torch.Tensor,
        depth_map: torch.Tensor = None,
        rgb_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """Samples images from the model for a given encoder_hidden_states (aka conditioning)"""
        sampled_latents = self.sampling_loop(
            posed_clip_embedding,
            vae_embedding,
            self.cfg.scheduler.num_inference_steps,
            depth_map,
            rgb_cond,
        )
        sampled_images = self.latent2img(sampled_latents)
        return sampled_images

    def plot_sampling_loop(
        self,
        pred,
        target,
        timesteps,
        image_target,
        image_cond,
        posed_clip_embedding,
        vae_embedding,
        depth_map,
        rgb_cond,
    ) -> None:
        logger = self.logger.experiment
        train_or_val = "Train" if self.training else "Val"

        # 1. Log target and conditionig image
        im_target = self.transform_decoded(image_target)
        im_cond = self.transform_decoded(image_cond)

        # logger.log({f"{train_or_val}/Target Image": wandb.Image(im_target)})  # type: ignore

        # 2. Log predicted image (to random step t and back) and corresponding v/epsilon target
        # loss in latent space from this
        pred_decoded = self.latent2img(pred)
        pred_img = self.transform_decoded(pred_decoded)
        latent_target_decoded = self.latent2img(target)
        latent_target = self.transform_decoded(latent_target_decoded)

        pred_visus = [
            wandb.Image(
                latent_target,
                caption=f"Target ({self.noise_scheduler.config.prediction_type})",
            ),
            wandb.Image(
                pred_img,
                caption=f"Prediction ({self.noise_scheduler.config.prediction_type})",
            ),
        ]

        logger.log({f"{train_or_val}/Unet Prediction": pred_visus})

        # logger.log({f"{train_or_val}/Prediction Image": wandb.Image(pred_img)})  # type: ignore

        # print("Timesteps: logging ", timesteps)
        logger.log({"Timestep for prediction": timesteps})

        # 3. Run sampling loop (from random noise) and log sample
        # metrics in image space from this
        sampled_batch = self.get_sampled_images(
            posed_clip_embedding, vae_embedding, depth_map, rgb_cond
        )
        sampled_batch = self.transform_decoded(sampled_batch, return_batch=True)

        # logger.log({f"{train_or_val}/Sample generations": wandb.Image(im)})  # type: ignore

        image_list = [
            wandb.Image(im_cond, caption="Conditioning image"),
            wandb.Image(im_target, caption="Target image"),
        ] + [
            wandb.Image(im, caption=f"Sample {i}") for i, im in enumerate(sampled_batch)
        ]
        if self.cfg.depth_conditioning.enable:
            # does not work for batch size > 1
            depth_map = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )
            depth_map = Image.fromarray(
                np.uint8(depth_map.squeeze().detach().cpu().numpy() * 255), "L"
            )
            image_list.append(wandb.Image(depth_map, caption="Depth map"))

        if self.cfg.rgb_conditioning.enable:
            rgb_cond = Image.fromarray(
                np.uint8(
                    rgb_cond.squeeze().detach().permute(1, 2, 0).cpu().numpy() * 255
                ),
                "RGB",
            )
            image_list.append(wandb.Image(rgb_cond, caption="RGB conditioning"))

        logger.log({f"{train_or_val}/Unet Sampling:": image_list})

        # 5. Compute and log FID
        # print(pred_decoded.dtype, target_decoded.dtype)
        # TODO: Fix: RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same
        # with torch.no_grad():
        #     self.fid.update(target_image, real=True)
        #     self.fid.update(sampled_image, real=False)
        #     fid = self.fid.compute()
        # logger.log({"Metrics/FID {train_or_val}": fid})

    @rank_zero_print_log_time
    def compute_recon_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute LPIPS, SSIM, PSNR metrics assuming image space
        pred and target in range [-1,1].
        TODO: make sure this works for batch size > 1
        """
        with torch.no_grad():
            lpips = self.lpips_loss(pred, target)
            ssim = self.ssim_loss(pred, target)
            psnr_value = calculate_psnr(pred, target)
        return (
            lpips.detach().cpu().item(),
            ssim.detach().cpu().item(),
            psnr_value.detach().cpu().item(),
        )

    def snr_loss(
        self, pred: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        # from https://github.com/huggingface/diffusers/blob/f72b28c75b2b4b720a5d8de78556694cf4b893fd/examples/dreambooth/train_dreambooth.py#L1281 # noqa
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(self.noise_scheduler, timesteps)
        # self.log("SNR", snr, on_step=True) not multi batch like this
        base_weight = (
            torch.stack(
                [snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
            ).min(dim=1)[0]
            / snr
        )

        if self.noise_scheduler.config.prediction_type == "v_prediction":
            # Velocity objective needs to be floored to an SNR weight of one.
            mse_loss_weights = base_weight + 1
        else:
            # Epsilon and sample both use the same loss weights.
            mse_loss_weights = base_weight
        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        return loss.mean()

    def depth_cutter(self, depth_map: torch.Tensor) -> torch.Tensor:
        # Simulate cutting
        # randomly sample value between 0.35 and 0.65
        cut_percentage = torch.rand(1) * 0.3 + 0.35
        width = depth_map.size(3)
        cut_index = int(width * cut_percentage)
        # 50/50 chance to cut left or right
        if torch.rand(1) > 0.5:
            depth_map[:, :, :, :cut_index] = 0
        else:
            depth_map[:, :, :, cut_index:] = 0

        # Repeat for height
        height = depth_map.size(2)
        cut_index = int(height * cut_percentage)
        # 50/50 chance to cut top or bottom
        if torch.rand(1) > 0.5:
            depth_map[:, :, :cut_index, :] = 0
        else:
            depth_map[:, :, cut_index:, :] = 0

        return depth_map

    def shared_step(self, batch, batch_idx):
        # shape [1,3,1920,1440]
        image_cond = batch["image_cond"]
        # shape [1,3,768,768] or [1,3,512,512]
        image_target = batch["image_target"]

        if self.cfg.depth_conditioning.enable:
            depth_map = batch["depth_map"]

        if self.cfg.rgb_conditioning.enable:
            rgb_cond = batch["rgb_cond"]

        T = batch["T"]
        path_cond = batch["path_cond"]
        image_cond_vae = batch["image_cond_vae"]
        b = image_cond.size(0)

        # cut depth map
        if self.cfg.depth_conditioning.enable:
            assert self.data_cfg.depth_map in [
                "partial_gt",
                "gt",
                "projected",
            ], "Depth conditioning only implemented for gt or projected depth maps, \
                make sure to set data_cfg.depth_map to one of those"
        if (
            self.data_cfg.depth_map == "partial_gt"
            and self.cfg.depth_conditioning.enable
        ):
            depth_map = self.depth_cutter(depth_map)

        # check if empty prompt is on gpu and if not move it there
        if not self.empty_prompt.is_cuda:
            self.empty_prompt = self.empty_prompt.to(self.device)
            # half precision
            self.empty_prompt = self.empty_prompt.half()

        # preprocess image
        image_cond_clip = self.feature_extractor_clip(
            images=image_cond, return_tensors="pt"
        ).pixel_values

        if self.image_embeds_to_disk:
            assert self.vision_encoder is None, "Vision encoder should be deleted"
            image_embeddings = torch.zeros(
                (b, 1, 1024), device=self.device, requires_grad=False
            )
            for i in range(b):
                # load image embeddings from disk for every image in batch
                directory, filename = os.path.split(path_cond[i])
                filename = os.path.splitext(filename)[0]
                parent_dir = os.path.dirname(directory)
                directory = os.path.join(parent_dir, "rgb_image_embeddings")
                load_path = os.path.join(directory, f"{filename}.pt")
                image_embeddings_i = torch.load(load_path)  # shape [1, 1024]
                assert image_embeddings_i.shape == torch.Size([1, 1024])
                image_embeddings[i] = image_embeddings_i
        else:
            # get encoder_hidden_states from CLIP vision model concat pose into projection
            image_embeddings = self.encode_image(image_cond_clip)  # shape: [1, 1024]
            # shape: [1, 1, 1024]
            image_embeddings = image_embeddings.unsqueeze(-2)

        # DEPTH Encoding
        # copy depth map to save original in own variable
        if self.cfg.depth_conditioning.enable:
            depth_map = self.encode_depth(depth_map)

        # Positional Encoding
        T = self.positional_encoding(T)  # shape: [1, 140]
        T = T.unsqueeze(-2)  # shape: [1, 1, 140]

        posed_clip_embedding = torch.cat(
            [image_embeddings, T], dim=-1
        )  # shape: [1, 1, 1024+140]

        posed_clip_embedding = self.pose_projection(
            posed_clip_embedding.half()
        )  # shape: [1, 1, 1024]
        posed_clip_embedding = posed_clip_embedding.expand(
            -1, 77, -1
        )  # shape: [1, 77, 1024]

        if self.cfg.flex_diffuse.enable:  # 2.4 Zero123++ Flex diffuse
            scaled_posed_clip_embedding = self.linear_flex_diffuse(posed_clip_embedding)
            posed_clip_embedding = scaled_posed_clip_embedding + self.empty_prompt
        else:
            posed_clip_embedding = posed_clip_embedding + self.empty_prompt

        if self.cfg.dreampose_adapter.enable:
            # ultra hacky to do the preprocessing here
            with torch.no_grad():
                vae_embedding = (
                    self.vae.encode(image_cond_vae).latent_dist.sample().half()
                )

                rank_zero_print(
                    "vae_embedding shape AFTER ENCODE: ", vae_embedding.shape
                )

        image_target = image_target.half()  # necessary ?
        posed_clip_embedding = posed_clip_embedding.half()  # necessary?

        unet_output, noise, timesteps, x0 = self.forward(
            image_target,
            posed_clip_embedding,
            vae_embedding if self.cfg.dreampose_adapter.enable else None,
            depth_map if self.cfg.depth_conditioning.enable else None,
            rgb_cond if self.cfg.rgb_conditioning.enable else None,
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

        if self.cfg.snr_gamma is None or self.cfg.snr_gamma == 0:
            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        else:
            loss = self.snr_loss(pred, target, timesteps)

        return {
            "loss": loss,
            "pred": pred,
            "image_target": image_target,
            "posed_clip_embedding": posed_clip_embedding,
            "vae_embedding": vae_embedding
            if self.cfg.dreampose_adapter.enable
            else None,
            "image_cond": image_cond_clip,
            "target": target,
            "timesteps": timesteps,
            "depth_map": depth_map if self.cfg.depth_conditioning.enable else None,
            "rgb_cond": rgb_cond if self.cfg.rgb_conditioning.enable else None,
        }

    def training_step(self, batch, batch_idx):
        """
        More options:
        # print(list(self.linear_flex_diffuse.parameters()))
        # print(list(self.pose_projection.parameters()))

        # For unet conditioning: Notes other inputs
        # added_cond_kwargs: (`dict`, *optional*):
        # A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
        # are passed along to the UNet blocks.
        # class_labels (`torch.Tensor`, *optional*, defaults to `None`):
        # Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        """

        shared_step_output = self.shared_step(batch, batch_idx)

        loss = shared_step_output["loss"]
        pred = shared_step_output["pred"]
        image_target = shared_step_output["image_target"]
        posed_clip_embedding = shared_step_output["posed_clip_embedding"]
        vae_embedding = shared_step_output["vae_embedding"]
        image_cond = shared_step_output["image_cond"]
        target = shared_step_output["target"]
        timesteps = shared_step_output["timesteps"]
        depth_map = shared_step_output["depth_map"]
        rgb_cond = shared_step_output["rgb_cond"]
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        if (
            self.train_iteration == 0
            and self.current_epoch % self.logger_cfg.log_metrics_train_every_n_epochs
            == 0
        ):
            sampled_images = self.get_sampled_images(
                posed_clip_embedding, vae_embedding, depth_map, rgb_cond
            )
            sampled_images = torch.clamp(sampled_images, -1, 1)  # For LPIPS
            image_target = torch.clamp(image_target, -1, 1)  # For LPIPS
            self.lpips_loss_train(sampled_images, image_target)
            self.ssim_loss_train(sampled_images, image_target)
            self.log("train_lpips", self.lpips_loss_train, on_epoch=True, on_step=False)
            self.log("train_ssim", self.ssim_loss_train, on_epoch=True, on_step=False)

        # log first sample of epoch
        if (
            self.train_iteration == 0
            and self.current_epoch % self.logger_cfg.log_image_every_n_epochs == 0
        ):
            self.plot_sampling_loop(
                pred[0].unsqueeze(0),
                target[0].unsqueeze(0),
                timesteps[0].unsqueeze(0),
                image_target[0].unsqueeze(0),
                image_cond[0].unsqueeze(0),
                posed_clip_embedding[0].unsqueeze(0),
                vae_embedding[0].unsqueeze(0)
                if self.cfg.dreampose_adapter.enable
                else None,
                depth_map[0].unsqueeze(0)
                if self.cfg.depth_conditioning.enable
                else None,
                rgb_cond[0].unsqueeze(0) if self.cfg.rgb_conditioning.enable else None,
            )

        self.train_iteration += 1

        return loss

    # TODO: change back again, only for overfitting debug here
    def validation_step(self, batch, batch_idx):
        # print out flex diffuse params

        shared_step_output = self.shared_step(batch, batch_idx)
        loss = shared_step_output["loss"]
        pred = shared_step_output["pred"]
        image_target = shared_step_output["image_target"]
        posed_clip_embedding = shared_step_output["posed_clip_embedding"]
        vae_embedding = shared_step_output["vae_embedding"]
        image_cond = shared_step_output["image_cond"]
        target = shared_step_output["target"]
        timesteps = shared_step_output["timesteps"]
        depth_map = shared_step_output["depth_map"]
        rgb_cond = shared_step_output["rgb_cond"]

        sampled_images = self.get_sampled_images(
            posed_clip_embedding, vae_embedding, depth_map, rgb_cond
        )
        sampled_images = torch.clamp(sampled_images, -1, 1)  # For LPIPS
        image_target = torch.clamp(image_target, -1, 1)  # For LPIPS
        # print shapes
        # rank_zero_print("Sampled images shape: ", sampled_images.shape)
        # rank_zero_print("Image target shape: ", image_target.shape)

        # print dtypes
        # rank_zero_print("Sampled images dtype: ", sampled_images.dtype)
        # rank_zero_print("Image target dtype: ", image_target.dtype)

        # print device
        # rank_zero_print("Sampled images device: ", sampled_images.device)
        # rank_zero_print("Image target device: ", image_target.device)

        self.lpips_loss_val.update(sampled_images, image_target)
        self.ssim_loss_val.update(sampled_images, image_target)
        # self.psnr_loss_val.update(sampled_images, image_target)

        if self.val_iteration == 0:
            self.plot_sampling_loop(
                pred[0].unsqueeze(0),
                target[0].unsqueeze(0),
                timesteps[0].unsqueeze(0),
                image_target[0].unsqueeze(0),
                image_cond[0].unsqueeze(0),
                posed_clip_embedding[0].unsqueeze(0),
                vae_embedding[0].unsqueeze(0)
                if self.cfg.dreampose_adapter.enable
                else None,
                depth_map[0].unsqueeze(0)
                if self.cfg.depth_conditioning.enable
                else None,
                rgb_cond[0].unsqueeze(0) if self.cfg.rgb_conditioning.enable else None,
            )

        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        self.val_iteration += 1

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss

    def on_train_epoch_end(self):
        self.train_iteration = 0
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        self.log("val_lpips", self.lpips_loss_val, on_epoch=True, on_step=False)
        self.log("val_ssim", self.ssim_loss_val, on_epoch=True, on_step=False)
        self.val_iteration = 0

    def configure_optimizers(self):
        optimizer_type = self.optimizer_dict.type
        higher_pp_lr = self.optimizer_dict.higher_pp_lr
        print("Higher pp lr: ", higher_pp_lr)

        if optimizer_type == "AdamW":
            if higher_pp_lr:
                pose_proj_params = list(self.pose_projection.parameters())
                unet_params = list(self.unet.parameters())
                if self.cfg.flex_diffuse.enable:
                    unet_params += list(self.linear_flex_diffuse.parameters())
                if self.cfg.depth_conditioning.enable:
                    unet_params += list(self.depth_feature_extractor.parameters())

                lr = self.optimizer_dict.params.lr
                del self.optimizer_dict.params.lr
                pp_lr = lr * 10
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": unet_params,
                            "lr": lr,
                            **self.optimizer_dict.params,
                        },
                        {
                            "params": pose_proj_params,
                            "lr": pp_lr,
                            **self.optimizer_dict.params,
                        },
                    ]
                )
            else:
                all_params = list(self.unet.parameters()) + list(
                    self.pose_projection.parameters()
                )
                # if self.cfg.depth_conditioning.enable:
                #    all_params += list(self.depth_feature_extractor.parameters())
                if self.cfg.flex_diffuse.enable:
                    all_params += list(self.linear_flex_diffuse.parameters())
                if self.cfg.dreampose_adapter.enable:
                    all_params += list(self.dreampose_adapter.parameters())
                optimizer = torch.optim.AdamW(
                    all_params,
                    **self.optimizer_dict.params,
                )
        else:
            raise ValueError(f"Unknown optimizer type {self.optimizer_dict.type}")

        return optimizer  # FP16_Optimizer(optimizer)
