import os
from typing import Optional, Tuple, Union

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from deepspeed.ops.adam import DeepSpeedCPUAdam
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from evaluation.metrics import calculate_psnr
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from PIL import Image

# from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
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

# from torch_ort.optim import FP16_Optimizer


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
                self.weights[0, i].fill_(i / L)
            rank_zero_print("Flex diffuse weights: ", self.weights)

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
        self.image_size = cfg.datamodule.image_size
        self.optimizer_dict = cfg.trainer.optimizer
        self.cfg = cfg.model
        self.enable_cfg = self.cfg.guidance.enable
        if self.enable_cfg:
            self.cfg_scale = self.cfg.guidance.cfg_scale
            rank_zero_print("CFG enabled with scale: ", self.cfg_scale)
        if self.cfg.snr_gamma != 0:
            rank_zero_print("SNR gamma enabled: ", self.cfg.snr_gamma)

        # metrics
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze"
        ).requires_grad_(False)
        self.ssim_loss = StructuralSimilarityIndexMeasure(
            data_range=(-1, 1), reduction="none"
        ).requires_grad_(False)
        self.fid = FrechetInceptionDistance(feature=64, normalize=False).requires_grad_(
            False
        )

        # core parts
        # self.feature_extractor_vae = CLIPImageProcessor.from_pretrained(
        #     self.cfg.feature_extractor_vae_path, subfolder="feature_extractor_vae"
        # )
        self.vae = AutoencoderKL.from_pretrained(
            self.cfg.vae.path, subfolder="vae", variant=self.cfg.vae.variant
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.cfg.unet.path,
            subfolder="unet",
            variant=self.cfg.unet.variant,
        )

        # initialize another fully-connected layer for posed CLIP embedding Appendix C Zero123
        self.pose_projection = torch.nn.Linear(1028, 1024)
        self.pose_projection.requires_grad_(True)

        # 2.4 Zero123++ Flex diffuse
        if self.cfg.flex_diffuse.enable:
            self.linear_flex_diffuse = LinearFlexDiffuse(self.cfg.flex_diffuse.L)
        # CLIP models for conditioning
        self.feature_extractor_clip = CLIPImageProcessor.from_pretrained(
            self.cfg.feature_extractor_clip_path, subfolder="feature_extractor"
        )
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
            if not os.path.exists(self.cfg.empty_prompt_embed_path):
                self.empty_prompt = self.encode_prompt("")
                empty_prompt_dir = os.path.dirname(self.cfg.empty_prompt_embed_path)
                os.makedirs(empty_prompt_dir, exist_ok=True)
                torch.save(self.empty_prompt, self.cfg.empty_prompt_embed_path)
                if not self.cfg.text_encoder.enable:
                    self.text_encoder = None
                    torch.cuda.empty_cache()

        if self.cfg.vision_encoder.enable:
            self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.cfg.vision_encoder.path, subfolder="image_encoder"
            )

        # get empty prompt embedding
        self.empty_prompt = torch.load(
            self.cfg.empty_prompt_embed_path, map_location=self.device
        )
        # rank_zero_print(
        #     "Empty prompt shape: ", self.empty_prompt.shape
        # )  # shape: [1, 77, 1024]
        # rank_zero_print("Empty prompt requires grad: ", self.empty_prompt.requires_grad)
        self.empty_prompt.requires_grad_(False)  # is false already

        # noise scheduler
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
        # TODO: store on disk if memory is an issue
        with torch.no_grad():
            embedding_from_projection = self.vision_encoder(image).image_embeds

        return embedding_from_projection

    def forward(self, image_target: torch.Tensor, encoder_hidden_states: torch.Tensor):
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

        if self.enable_cfg:
            # with probability 0.2 set the conditioning to empty prompt
            if torch.rand(1) < 0.2:
                encoder_hidden_states = self.empty_prompt  # .half().to(self.device)

        # rank_zero_print("x0 shape (after noise addition): ", noisy_x0.shape)
        # Get the model prediction
        unet_output = self.unet(
            sample=noisy_x0,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )

        self.log("timesteps", timesteps.float())

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
        height = (
            self.image_size // 8
        )  # self.unet.config.sample_size #64 if 512, 96 if 768
        width = (
            self.image_size // 8
        )  # self.unet.config.sample_size #64 if 512, 96 if 768
        # generate random latents
        latents = torch.randn(
            (1, num_channels_latents, height, width),
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
                [self.empty_prompt, encoder_hidden_states]
            )
            encoder_hidden_states = encoder_hidden_states.half()  # why again ?

        # plotting_timesteps = []
        # plotting_latents = []
        # plotting_pred_original_sample = []
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

    def transform_latent(
        self,
        image: torch.Tensor,
        return_PIL: Optional[bool] = True,
        return_batch: Optional[bool] = False,
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Convert first image from tensor to PIL image or tensor for visualization
        """
        if return_batch:
            assert return_PIL is False
        image = torch.clamp(image, -1, 1)
        image = (image + 1) / 2
        image = image.permute(0, 2, 3, 1)
        image = image.detach().cpu()
        image = (image * 255).int()

        if return_PIL:
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
            decoded_latents = self.transform_latent(
                self.latent2img(latents_i), return_PIL=False, return_batch=True
            )
            grid = torchvision.utils.make_grid(decoded_latents, nrow=1)
            axs[i, 0].imshow(grid.cpu())
            axs[i, 0].set_title(f"Current x (step {timesteps[i]})")

            pred_x0 = pred_original_sample_i
            pred_x0 = self.transform_latent(
                self.latent2img(pred_x0), return_PIL=False, return_batch=True
            )
            grid = torchvision.utils.make_grid(pred_x0, nrow=1)
            axs[i, 1].imshow(grid.cpu())
            axs[i, 1].set_title(f"Predicted denoised images (step {timesteps[i]})")

        self.logger.experiment.log(
            {f"{train_or_val}/Sampling Loop": wandb.Image(fig)}
        )  # type: ignore

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
        train_or_val = "Train" if self.training else "Val"

        # 1. Log target and conditionig image
        im_target = self.transform_latent(image_target)
        im_cond = self.transform_latent(image_cond)

        logger.log({f"{train_or_val}/Target Image": wandb.Image(im_target)})  # type: ignore

        logger.log({f"{train_or_val}/Conditioning Image": wandb.Image(im_cond)})  # type: ignore

        # 2. Log predicted image (to random step t and back)
        # loss in latent space from this
        pred_decoded = self.latent2img(pred)
        pred_img = self.transform_latent(pred_decoded)

        logger.log({f"{train_or_val}/Prediction Image": wandb.Image(pred_img)})  # type: ignore
        # target_img = self.transform_latent(target_decoded)  # TODO: remove, we don't need this
        # logger.log({f"{train_or_val}/Target Image": wandb.Image(target_img)})  # type: ignore
        # print("Timesteps: logging ", timesteps)
        logger.log({"Timestep for prediction": timesteps})

        # 3. Run sampling loop (from random noise) and log sample
        # metrics in image space from this
        num_inference_steps = self.cfg.scheduler.num_inference_steps
        latents = self.sampling_loop(encoder_hidden_states, num_inference_steps)

        sampled_image = self.latent2img(latents)
        im = self.transform_latent(sampled_image)

        logger.log({f"{train_or_val}/Sample generations": wandb.Image(im)})  # type: ignore

        # 4. Compute and log reconstruction and quality metrics
        sampled_image = torch.clamp(sampled_image, -1, 1)
        target_image = torch.clamp(image_target, -1, 1)
        lpips, ssim, psnr = self.compute_recon_metrics(sampled_image, target_image)
        logger.log({f"Metrics/LPIPS {train_or_val}": lpips})
        logger.log({f"Metrics/SSIM {train_or_val}": ssim})
        logger.log({f"Metrics/PSNR {train_or_val}": psnr})

        # delete everything from gpu
        # del latents
        # del pred_decoded
        # del pred_img
        # del im_target
        # del im_cond
        # del im
        # del sampled_image
        # del target_image
        # del lpips
        # del ssim
        # del psnr

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

    def shared_step(self, batch, batch_idx):
        # .to(self.device)  # shape [1,3,1920,1440]
        image_cond = batch["image_cond"]
        # .to(self.device)# shape [1,3,768,768] or [1,3,512,512]
        image_target = batch["image_target"]
        T = batch["T"]

        # check if empty prompt is on gpu and if not move it there
        if not self.empty_prompt.is_cuda:
            self.empty_prompt = self.empty_prompt.to(self.device)
            # half precision
            self.empty_prompt = self.empty_prompt.half()

        # rank_zero_print("Image Cond Shape", image_cond.shape)
        # rank_zero_print("Image Target Shape", image_target.shape)

        # preprocess image
        image_cond = self.feature_extractor_clip(
            images=image_cond, return_tensors="pt"
        ).pixel_values

        # rank_zero_print(
        #     "Image Cond Shape after feature extractor clip", image_cond.shape
        # )

        # image_target = self.feature_extractor_vae(
        #    images=image_target, return_tensors="pt"
        # ).pixel_values

        # because feature_extractor removes from gpu ?
        image_cond = image_cond.to(self.device)
        # image_target = image_target#.to(self.device)

        # get encoder_hidden_states from CLIP vision model concat pose into projection
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
        posed_clip_embedding = posed_clip_embedding.expand(
            -1, 77, -1
        )  # shape: [1, 77, 1024]

        if self.cfg.flex_diffuse.enable:  # 2.4 Zero123++ Flex diffuse
            scaled_posed_clip_embedding = self.linear_flex_diffuse(posed_clip_embedding)
            encoder_hidden_states = (
                scaled_posed_clip_embedding + self.empty_prompt  # .to(self.device)
            )
        else:
            encoder_hidden_states = posed_clip_embedding

        # print types and shapes
        # rank_zero_print("image_cond: ", image_cond.dtype, image_cond.shape)
        # rank_zero_print("image_target: ", image_target.dtype, image_target.shape)
        # rank_zero_print(
        #     "encoder_hidden_states: ",
        #     encoder_hidden_states.dtype,
        #     encoder_hidden_states.shape,
        # )

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

        # print("loss: ", F.mse_loss(pred.float(), target.float(), reduction="mean"))
        # from https://github.com/huggingface/diffusers/blob/f72b28c75b2b4b720a5d8de78556694cf4b893fd/examples/dreambooth/train_dreambooth.py#L1281 # noqa
        if self.cfg.snr_gamma is None or self.cfg.snr_gamma == 0:
            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            self.log("SNR", snr, on_step=True)
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
            loss = loss.mean()

        self.log("train_loss", loss, on_epoch=True, on_step=True)

        # log first sample of epoch
        if (
            self.train_iteration == 0
            and self.current_epoch % self.logger_cfg.log_image_every_n_epochs == 0
        ):
            self.plot_sampling_loop(
                pred,
                target,
                timesteps,
                image_target,
                image_cond,
                encoder_hidden_states,
            )
            self.train_iteration += 1

        return loss

    # TODO: change back again, only for overfitting debug here
    def validation_step(self, batch, batch_idx):
        return 0.0
        # loss = self.training_step(batch, batch_idx)

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

        if self.cfg.snr_gamma is None or self.cfg.snr_gamma == 0:
            loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            self.log("SNR", snr, on_step=True)
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
            loss = loss.mean()

        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        if (
            self.val_iteration == 0
            and self.current_epoch % self.logger_cfg.log_image_every_n_epochs == 0
        ):
            self.plot_sampling_loop(
                pred,
                target,
                timesteps,
                image_target,
                image_cond,
                encoder_hidden_states,
            )
            self.val_iteration += 1

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss

    def on_train_epoch_end(self):
        # else it will accumulate over epochs see https://github.com/Lightning-AI/pytorch-lightning/issues/5733
        self.ssim_loss.reset()
        self.train_iteration = 0
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        self.ssim_loss.reset()
        self.val_iteration = 0

    def configure_optimizers(self):
        # all_params = list(self.unet.parameters())+list(self.linear_flex_diffuse.parameters())#,self.pose_projection.parameters())#
        optimizer_type = self.optimizer_dict.type
        # all_params = self.unet.parameters()
        all_params = list(self.unet.parameters())
        all_params += list(self.pose_projection.parameters())
        if self.cfg.flex_diffuse.enable:
            all_params += list(self.linear_flex_diffuse.parameters())

        if optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                all_params,
                **self.optimizer_dict.params,
            )
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                all_params,
                **self.optimizer_dict.params,
            )
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                all_params,
                **self.optimizer_dict.params,
            )
        elif optimizer_type == "DeepSpeedCPUAdam":
            optimizer = DeepSpeedCPUAdam(all_params, **self.optimizer_dict.params)
        else:
            raise ValueError(f"Unknown optimizer type {self.optimizer_dict.type}")

        return optimizer  # FP16_Optimizer(optimizer)
