# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
# limitations under the License.

"""IP2P_PTD module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from typing import Union

import torch
from rich.console import Console
from torch import Tensor, nn
from torch.nn import functional as F
from jaxtyping import Float
import numpy as np
from tqdm import tqdm

CONSOLE = Console(width=120)

# sd utils
from utils.sds_utils import (
    encode_prompt_with_a_prompt_and_n_prompt,
) 

try:
    from diffusers import (
        DDIMScheduler,
        StableDiffusionInstructPix2PixPipeline,
        ControlNetModel, 
        StableDiffusionControlNetPipeline
    )
    from transformers import logging

except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"

CN_SOURCE = "lllyasviel/control_v11f1p_sd15_depth"


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class IP2P_PTD(nn.Module):
    """IP2P_PTD implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(
        self, 
        dtype,
        device: Union[torch.device, str], 
        conditioning_scale: float = 1.0,
        prompt: str = "a photo of a japanese style living room",
        t_dec: int = 20, 
        num_train_timesteps: int = 1000, 
        ip2p_use_full_precision=False,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        render_size: int = 512,
        blending_ratio_default: float = 1.0,
        direct_transfer_ratio: float = 0.4,
        decayed_transfer_ratio: float = 0.2,
        exponent: float = 0.5,
        async_ahead_steps: int = 0,
    ) -> None:
        
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.conditioning_scale = conditioning_scale
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.blending_ratio_default = blending_ratio_default
        self.direct_transfer_ratio = direct_transfer_ratio
        self.decayed_transfer_ratio = decayed_transfer_ratio
        self.exponent = exponent
        self.render_size = render_size
        self.async_ahead_steps = async_ahead_steps

        pipe_ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        pipe_ip2p.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe_ip2p.scheduler.set_timesteps(100)
        assert pipe_ip2p is not None
        pipe_ip2p = pipe_ip2p.to(self.device)

        self.pipe_ip2p = pipe_ip2p

        # improve memory performance
        pipe_ip2p.enable_attention_slicing()

        self.scheduler_ip2p = pipe_ip2p.scheduler
        self.alphas_ip2p = self.scheduler_ip2p.alphas_cumprod.to(self.device)  # type: ignore

        pipe_ip2p.unet.eval()
        pipe_ip2p.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe_ip2p.unet.float()
            pipe_ip2p.vae.float()
        else:
            if self.device.index:
                pipe_ip2p.enable_model_cpu_offload(self.device.index)
            else:
                pipe_ip2p.enable_model_cpu_offload(0)

        self.unet_ip2p = pipe_ip2p.unet
        self.vae_ip2p = pipe_ip2p.vae

        CONSOLE.print("IP2P loaded!")

        # PTD configs
        self.batch_size = 1
        self.prompt = prompt
        self.a_prompt = ", best quality, high quality, extremely detailed, good geometry, high-res photo"
        self.n_prompt = "deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke, shading, lighting, lumination, shadow, text in image, watermarks"
        # load model
        controlnet = ControlNetModel.from_pretrained(CN_SOURCE, torch_dtype=self.dtype, low_cpu_mem_usage=True).to(self.device)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(SD_SOURCE, controlnet=controlnet, torch_dtype=self.dtype, low_cpu_mem_usage=True).to(self.device)
        pipe.enable_model_cpu_offload()
        controlnet = pipe.controlnet.to(self.dtype)
        controlnet.requires_grad_(False)
        # components
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet.to(self.dtype)
        # save VRAM
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
        # set device
        vae = vae.to(self.device).requires_grad_(False)
        text_encoder = text_encoder.to(self.device).requires_grad_(False)
        unet = unet.to(self.device).requires_grad_(False)
        # set scheduler
        scheduler = DDIMScheduler.from_pretrained(SD_SOURCE, subfolder="scheduler", torch_dtype=self.dtype)
        scheduler.betas = scheduler.betas.to(self.device)
        scheduler.alphas = scheduler.alphas.to(self.device)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        # set timesteps
        num_train_timesteps = len(scheduler.betas)
        scheduler.set_timesteps(num_train_timesteps)

        # set self parameters
        self.pipe_ptd = pipe
        self.controlnet_ptd = controlnet
        self.tokenizer_ptd = tokenizer
        self.text_encoder_ptd = text_encoder
        self.vae_ptd = vae
        self.unet_ptd = unet
        self.scheduler_ptd = scheduler

        CONSOLE.print("PTDiffusion loaded!")

        # ref latent
        # self.ref_latent_path = './outputs/latent.pt'
        # self.ref_latent_path = './outputs/latent_face1.jpg_PTD.pt'
        # self.ref_latent_path = './outputs/latent_face1.jpt_cond.pt' # japanese
        # self.ref_latent_path = './outputs/latent_face1.jpg_cond.pt' # snowed 

        # don't use cond for DDIM inversion above, the results is just not stable
        self.ref_latent_path = 'latent_face1.jpg.pt'
        # self.ref_latent_path = 'latent_face2.jpg.pt'
        # self.ref_latent_path = 'latent_tum_white.png.pt'
        # self.ref_latent_path = 'latent_yellow_dog.jpg.pt'

        # self.ref_latent_path_2 = 'latent_face1.jpg.pt'
        self.ref_latent_path_2 = 'latent_face2.jpg.pt'
        # self.ref_latent_path_2 = 'latent_tum_white.png.pt'
        # self.ref_latent_path_2 = 'latent_yellow_dog.jpg.pt'

        self.ref_latent_init = torch.load(self.ref_latent_path).cuda().to(self.dtype)

        self.ref_latent_2_init = torch.load(self.ref_latent_path_2).cuda().to(self.dtype)

        self.t_dec = t_dec

        # get the text embedding
        text_embeddings = encode_prompt_with_a_prompt_and_n_prompt(self.batch_size, self.prompt, self.a_prompt, self.n_prompt, tokenizer, text_encoder, self.device, particle_num_vsd=1)
        self.uncond, self.cond = text_embeddings.chunk(2)
        self.text_embeddings_ptd = text_embeddings

        self.text_embeddings_ip2p = torch.cat([self.cond, self.uncond, self.uncond])


    @torch.no_grad()
    def phase_substitute(
        self,
        ref_latent: torch.FloatTensor,
        x_dec: torch.FloatTensor,
        alpha: float = 0.0
    ) -> torch.FloatTensor:
        """Substitute the phase of x_dec with ref_latent according to blending ratio alpha.
        Args:
            ref_latent: reference latent
            x_dec: current latent
            alpha: blending ratio
        Returns:
            output: phase transferred latent
        """
        # move to CPU for stable FFT and back
        ref_cpu = ref_latent.to(torch.float32).cpu()
        x_cpu = x_dec.to(torch.float32).cpu()

        # FFTs
        ref_fft = torch.fft.fft2(ref_cpu, dim=(-2, -1)).to(x_dec.device) # must move the variables to cpu or FFT internal error, this happends for 4090, Ada and cuda-11.7, try to use 11.8.
        x_fft = torch.fft.fft2(x_cpu,     dim=(-2, -1)).to(x_dec.device)

        # ref_fft = torch.fft.fft2(ref_latent.to(torch.float16), dim=(-2, -1)).to(x_dec.device) # must move the variables to cpu or FFT internal error, this happends for 4090, Ada and cuda-11.7, try to use 11.8.
        # x_fft = torch.fft.fft2(x_dec.to(torch.float16),     dim=(-2, -1)).to(x_dec.device)

        # magnitudes and angles
        mag_x = torch.abs(x_fft)
        angle_ref = torch.angle(ref_fft)
        angle_x   = torch.angle(x_fft)

        # mix angles
        angle_mixed = angle_ref * alpha + angle_x *  (1 - alpha)

        # combined = mag_x * torch.exp(1j * angle_mixed)
        x_dec_fft = mag_x * torch.cos(angle_mixed) + mag_x * torch.sin(angle_mixed) * torch.complex(torch.zeros_like(mag_x), torch.ones_like(mag_x))

        # inverse FFT and restore to device
        output = torch.fft.ifft2(x_dec_fft.cpu()).real.to(x_dec.device)

        return output.to(self.dtype)

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x: torch.FloatTensor,
        cond: torch.FloatTensor,
        t: torch.LongTensor,
        index: int,
        last_index: int,
        image_cond_latents,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: torch.FloatTensor = None,
        return_all: bool = False,
    ) -> torch.FloatTensor:
        """
        Single DDIM inversion/sample step without any `self` references.
        Uses the scheduler's built-in alpha/beta arrays.
        """
        b, _, h, w = x.shape
        device = x.device
        x = x.to(self.dtype)

        # pull precomputed arrays directly from scheduler
        alphas_cumprod = self.scheduler_ptd.alphas_cumprod.to(device)
        one_minus_alphas_sqrt = (1.0 - alphas_cumprod).sqrt()
        sigmas = torch.zeros_like(alphas_cumprod, device=device)

        # gather scalars for this step
        a_t       = alphas_cumprod[index]
        a_prev    = alphas_cumprod[last_index]
        sigma_t   = sigmas[index]
        sqrt_1ma  = one_minus_alphas_sqrt[index]

        # reshape to match x
        a_t      = a_t.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        a_prev   = a_prev.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        sigma_t  = sigma_t.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        sqrt_1ma = sqrt_1ma.view(1, 1, 1, 1).expand(b, -1, -1, -1)

        # predict epsilon
        if unconditional_guidance_scale != 1.0 and unconditional_conditioning is not None:
            latent_model_input = torch.cat([x] * 3)
            latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
            out = self.unet_ip2p(latent_model_input, t, encoder_hidden_states=self.text_embeddings_ip2p).sample
            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = out.chunk(3)
            e_t = (
                noise_pred_uncond
                + unconditional_guidance_scale * (noise_pred_text - noise_pred_image)
                + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
        else:
            e_t = self.unet_ptd(x, t, encoder_hidden_states=cond).sample

        # predict x0
        pred_x0 = (x - sqrt_1ma * e_t) / a_t.sqrt()

        if return_all:
            return a_prev, pred_x0, e_t

        # compute x_{t-1}
        dir_xt = torch.sqrt(1 - a_prev - sigma_t.pow(2)) * e_t
        noise = sigma_t * torch.randn_like(x)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev.to(self.dtype)

    def edit_image(
        self,
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        secret_idx: int, # the index of the secret view, 0 for first view, 1 for second view
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using IP2P_PTD
        Args:
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        # reset the ref latent to prevent reusing the results from last edition 
        if secret_idx == 0:
            self.ref_latent = self.ref_latent_init.clone()
        if secret_idx == 1:
            self.ref_latent = self.ref_latent_2_init.clone()
        # linear ref timesteps
        ref_update_steps = np.linspace(990, 0, self.t_dec)

        weights_decay_transfer = torch.linspace(self.blending_ratio_default, 0.0, int(self.decayed_transfer_ratio * self.t_dec)) ** self.exponent

        self.stage = "direct_transfer"

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        # change the default value from 1000 to T.item(), then 
        # the chosen t would be in [0, T.item()] instead of [0, 1000].
        # To my experience, a larger T.item() would lead to a better result, 
        # since the model is trained with 1000 steps (a larger t span).
        self.scheduler_ip2p.config.num_train_timesteps = T.item() 
        self.scheduler_ip2p.set_timesteps(self.t_dec)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        noise = torch.randn_like(latents)
        # x_dec = self.scheduler_ip2p.add_noise(latents, noise, self.scheduler_ip2p.timesteps[0])  # type: ignore
        t_init = torch.tensor(int(ref_update_steps[0])).to(self.device)
        x_dec = self.scheduler_ptd.add_noise(latents, noise, t_init)

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        # loop = tqdm(range(self.t_dec))
        for i in range(self.t_dec):

            # # predict the noise residual with unet, NO grad!
            # with torch.no_grad():
            #     # pred noise
            #     latent_model_input = torch.cat([latents] * 3)
            #     latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

            #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # # perform classifier-free guidance
            # noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            # noise_pred = (
            #     noise_pred_uncond
            #     + guidance_scale * (noise_pred_text - noise_pred_image)
            #     + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            # )

            # # get previous sample, continue loop
            # latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            chosen_t = int(ref_update_steps[i])
            t = torch.tensor([chosen_t]).to(self.device)

            chosen_t_last = int(ref_update_steps[i + 1]) if i < self.t_dec - 1 else int(ref_update_steps[i])
            t_last = torch.tensor([chosen_t_last]).to(self.device)

            ts_tensor = torch.full((self.ref_latent.size(0),), chosen_t, device=self.ref_latent.device, dtype=torch.long) # 990 ~ 0
            
            ref_a_prev, ref_pred_x0, ref_e = self.p_sample_ddim(
                self.ref_latent, self.uncond, ts_tensor, t, t_last,
                image_cond_latents, return_all=True
            )
            dec_a_prev, dec_pred_x0, dec_e = self.p_sample_ddim(
                x_dec, self.cond, ts_tensor, t, t_last,
                image_cond_latents,
                unconditional_guidance_scale=self.guidance_scale,
                unconditional_conditioning=self.uncond,
                return_all=True
            )
            # reconstruct
            x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e

            # direct
            # self.ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e
             # async ahead reconstruction
            alphas_cumprod = self.scheduler_ptd.alphas_cumprod.to(self.device)
            if t_last - self.async_ahead_steps >= 0 and t_last - self.async_ahead_steps <= 999:
                ref_a_prev_ahead = alphas_cumprod[t_last - self.async_ahead_steps].view(1, 1, 1, 1).expand(self.batch_size, -1, -1, -1)
            else: 
                ref_a_prev_ahead = ref_a_prev
            self.ref_latent = ref_a_prev_ahead.sqrt() * ref_pred_x0 + (1 - ref_a_prev_ahead).sqrt() * ref_e

            # transfer latents
            if i <= (self.direct_transfer_ratio) * self.t_dec:
                self.stage = "direct_transfer"
                blending_ratio = self.blending_ratio_default
            elif i < (self.direct_transfer_ratio + self.decayed_transfer_ratio) * self.t_dec:
                self.stage = "decayed_transfer"
                blending_ratio = weights_decay_transfer[i - int((self.direct_transfer_ratio) * self.t_dec) - 1]
            else:
                self.stage = "refining"
                blending_ratio = 0.0

            # phase transfer
            x_dec = self.phase_substitute(self.ref_latent, x_prev, blending_ratio)

            # loop.set_description(f"Stage: {self.stage}, Step: {i + 1}/{self.t_dec}, Blending Ratio: {blending_ratio:.4f}")

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(x_dec)

        return decoded_img
    
    @torch.no_grad()
    def p_sample_ddim_depth(
        self,
        x: torch.FloatTensor,
        cond: torch.FloatTensor,
        t: torch.LongTensor,
        index: int,
        last_index: int,
        controlnet_cond_input: torch.FloatTensor,
        image_cond_latents,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: torch.FloatTensor = None,
        return_all: bool = False,
    ) -> torch.FloatTensor:
        """
        Single DDIM inversion/sample step without any `self` references.
        Uses the scheduler's built-in alpha/beta arrays.
        """
        b, _, h, w = x.shape
        device = x.device
        x = x.to(self.dtype)

        # pull precomputed arrays directly from scheduler
        alphas_cumprod = self.scheduler_ptd.alphas_cumprod.to(device)
        one_minus_alphas_sqrt = (1.0 - alphas_cumprod).sqrt()
        sigmas = torch.zeros_like(alphas_cumprod, device=device)

        # gather scalars for this step
        a_t       = alphas_cumprod[index]
        a_prev    = alphas_cumprod[last_index]
        sigma_t   = sigmas[index]
        sqrt_1ma  = one_minus_alphas_sqrt[index]

        # reshape to match x
        a_t      = a_t.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        a_prev   = a_prev.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        sigma_t  = sigma_t.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        sqrt_1ma = sqrt_1ma.view(1, 1, 1, 1).expand(b, -1, -1, -1)

        # predict epsilon
        if unconditional_guidance_scale != 1.0 and unconditional_conditioning is not None:
            latent_input = torch.cat([x] * 3)
            latent_model_input = torch.cat([latent_input, image_cond_latents], dim=1)
            unet_cross_attention_kwargs = {'scale': 0}
            controlnet_output = self.controlnet_ptd(latent_input.to(self.dtype), t, encoder_hidden_states=self.text_embeddings_ip2p, controlnet_cond=controlnet_cond_input, conditioning_scale=self.conditioning_scale, guess_mode=False, return_dict=True)
            out = self.unet_ip2p(latent_model_input, t, encoder_hidden_states=self.text_embeddings_ip2p, cross_attention_kwargs=unet_cross_attention_kwargs, down_block_additional_residuals=controlnet_output.down_block_res_samples, mid_block_additional_residual=controlnet_output.mid_block_res_sample).sample
            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = out.chunk(3)
            e_t = (
                noise_pred_uncond
                + unconditional_guidance_scale * (noise_pred_text - noise_pred_image)
                + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
        else:
            e_t = self.unet_ptd(x, t, encoder_hidden_states=cond).sample

        # predict x0
        pred_x0 = (x - sqrt_1ma * e_t) / a_t.sqrt()

        if return_all:
            return a_prev, pred_x0, e_t

        # compute x_{t-1}
        dir_xt = torch.sqrt(1 - a_prev - sigma_t.pow(2)) * e_t
        noise = sigma_t * torch.randn_like(x)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev.to(self.dtype)

    def get_depth_tensor(self, depth: Float[Tensor, "BS H W"]): # [0, 1], meter
        """Get depth tensor for image editing
        Args:
            depth: depth map
        Returns:
            normalized depth tensor
        """
        no_depth = 0.0
        pad_value = 10

        depth_min, depth_max = depth[depth != no_depth].min(), depth[depth != no_depth].max()
        target_min, target_max = 50, 255

        depth_value = depth[depth != no_depth]
        depth_value = depth_max - depth_value # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = depth.clone()
        relative_depth[depth != no_depth] = depth_value
        relative_depth[depth == no_depth] = pad_value # not completely black
        
        rel_depth_normalized = relative_depth.unsqueeze(1).to(self.device)
        assert len(rel_depth_normalized.shape) == 4
        rel_depth_normalized = rel_depth_normalized.repeat(1, 3, 1, 1).float()
        rel_depth_normalized = F.interpolate(rel_depth_normalized, (self.render_size, self.render_size), mode='bilinear', align_corners=False)
        # expected range [0, 1]
        rel_depth_normalized = rel_depth_normalized / 255.0
        depth_tensor = rel_depth_normalized.to(self.device).to(self.dtype)

        return depth_tensor

    def edit_image_depth(
        self,
        image: Float[Tensor, "BS 3 H W"],
        image_cond: Float[Tensor, "BS 3 H W"],
        secret_idx: int, # the index of the secret view, 0 for first view, 1 for second view
        depth: Float[Tensor, "BS H W"], # [0, 1], meter
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using IP2P_PTD
        Args:
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        # prepare depth condition input
        depth_tensor = self.get_depth_tensor(depth)
        controlnet_cond_input = torch.cat([depth_tensor] * 3)

        # reset the ref latent to prevent reusing the results from last edition 
        if secret_idx == 0:
            self.ref_latent = self.ref_latent_init.clone()
        if secret_idx == 1:
            self.ref_latent = self.ref_latent_2_init.clone()
        # linear ref timesteps
        ref_update_steps = np.linspace(990, 0, self.t_dec)

        weights_decay_transfer = torch.linspace(self.blending_ratio_default, 0.0, int(self.decayed_transfer_ratio * self.t_dec)) ** self.exponent

        self.stage = "direct_transfer"

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        # change the default value from 1000 to T.item(), then 
        # the chosen t would be in [0, T.item()] instead of [0, 1000].
        # To my experience, a larger T.item() would lead to a better result, 
        # since the model is trained with 1000 steps (a larger t span).
        self.scheduler_ip2p.config.num_train_timesteps = T.item() 
        self.scheduler_ip2p.set_timesteps(self.t_dec)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        noise = torch.randn_like(latents)
        # x_dec = self.scheduler_ip2p.add_noise(latents, noise, self.scheduler_ip2p.timesteps[0])  # type: ignore
        t_init = torch.tensor(int(ref_update_steps[0])).to(self.device)
        x_dec = self.scheduler_ptd.add_noise(latents, noise, t_init)

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        # loop = tqdm(range(self.t_dec))
        # for i in loop:

        for i in range(self.t_dec):

            # # predict the noise residual with unet, NO grad!
            # with torch.no_grad():
            #     # pred noise
            #     latent_model_input = torch.cat([latents] * 3)
            #     latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

            #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # # perform classifier-free guidance
            # noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            # noise_pred = (
            #     noise_pred_uncond
            #     + guidance_scale * (noise_pred_text - noise_pred_image)
            #     + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            # )

            # # get previous sample, continue loop
            # latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            chosen_t = int(ref_update_steps[i])
            t = torch.tensor([chosen_t]).to(self.device)

            chosen_t_last = int(ref_update_steps[i + 1]) if i < self.t_dec - 1 else int(ref_update_steps[i])
            t_last = torch.tensor([chosen_t_last]).to(self.device)

            ts_tensor = torch.full((self.ref_latent.size(0),), chosen_t, device=self.ref_latent.device, dtype=torch.long) # 990 ~ 0
            
            ref_a_prev, ref_pred_x0, ref_e = self.p_sample_ddim_depth(
                self.ref_latent, self.uncond, ts_tensor, t, t_last,
                controlnet_cond_input, image_cond_latents, return_all=True
            )
            dec_a_prev, dec_pred_x0, dec_e = self.p_sample_ddim_depth(
                x_dec, self.cond, ts_tensor, t, t_last,
                controlnet_cond_input, image_cond_latents,
                unconditional_guidance_scale=self.guidance_scale,
                unconditional_conditioning=self.uncond,
                return_all=True
            )
            # reconstruct
            x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e

            # direct
            # self.ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e
             # async ahead reconstruction
            alphas_cumprod = self.scheduler_ptd.alphas_cumprod.to(self.device)
            if t_last - self.async_ahead_steps >= 0 and t_last - self.async_ahead_steps <= 999:
                ref_a_prev_ahead = alphas_cumprod[t_last - self.async_ahead_steps].view(1, 1, 1, 1).expand(self.batch_size, -1, -1, -1)
            else: 
                ref_a_prev_ahead = ref_a_prev
            self.ref_latent = ref_a_prev_ahead.sqrt() * ref_pred_x0 + (1 - ref_a_prev_ahead).sqrt() * ref_e

            # transfer latents
            if i <= (self.direct_transfer_ratio) * self.t_dec:
                self.stage = "direct_transfer"
                blending_ratio = self.blending_ratio_default
            elif i < (self.direct_transfer_ratio + self.decayed_transfer_ratio) * self.t_dec:
                self.stage = "decayed_transfer"
                blending_ratio = weights_decay_transfer[i - int((self.direct_transfer_ratio) * self.t_dec) - 1]
            else:
                self.stage = "refining"
                blending_ratio = 0.0

            # phase transfer
            x_dec = self.phase_substitute(self.ref_latent, x_prev, blending_ratio)

            # loop.set_description(f"Stage: {self.stage}, Step: {i + 1}/{self.t_dec}, Blending Ratio: {blending_ratio:.4f}")

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(x_dec)

        return decoded_img, depth_tensor

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.vae_ptd.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.vae_ptd.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.vae_ptd.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
