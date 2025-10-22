import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
from jaxtyping import Float

from rich.console import Console

# diffusers for loading sd model
from diffusers import ( 
    DDIMScheduler, 
    ControlNetModel, 
    StableDiffusionControlNetPipeline
)

# sd utils
from utils.sds_utils import (
    encode_prompt_with_a_prompt_and_n_prompt,
) 

from transformers import AutoProcessor, AutoTokenizer, CLIPModel, CLIPVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

CONST_SCALE = 0.18215

CLIP_SOURCE = "openai/clip-vit-large-patch14"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CN_SOURCE = "lllyasviel/control_v11f1p_sd15_depth"

CONSOLE = Console(width=120)

def encode_image(image_original, dtype, device):
    # First option
    # # get the image embedding
    # image_encoder = CLIPVisionModel.from_pretrained(CLIP_SOURCE)
    # image_processor = CLIPImageProcessor.from_pretrained(CLIP_SOURCE)
    # projection = torch.nn.Linear(1024, 768).to(dtype)
    # # projection = torch.nn.Linear(1024, 768 * 4).to(self.dtype)

    # # # Load IP-Adapter weights
    # # ip_adapter_path = "ip_adapter/ip-adapter_sd15.safetensors"
    # # ip_adapter_state_dict = load_file(ip_adapter_path)

    # # projection.load_state_dict({
    # #     "weight": ip_adapter_state_dict["image_proj.proj.weight"],
    # #     "bias": ip_adapter_state_dict["image_proj.proj.bias"]
    # # })

    # inputs = image_processor(images=image_original, return_tensors="pt")
    # image_embeds = image_encoder(**inputs).last_hidden_state.to(dtype) # [1, 257, 1024]
    # conditional_embeds = projection(image_embeds) # [1, 257, 768]
    # # image_proj = projection(image_embeds)
    # # image_proj = image_proj.reshape(image_proj.shape[0], image_proj.shape[1], 4, 768)
    # # cls_embed = image_proj[:, 0, :, :]
    # # # conditional_embeds = cls_embed

    # unconditional_embeds = torch.zeros_like(conditional_embeds)
    # image_embeddings = torch.cat([unconditional_embeds, image_embeds]).to(device)

    # second option
    # ref: https://huggingface.co/openai/clip-vit-large-patch14/discussions/1
    _model = CLIPModel.from_pretrained(CLIP_SOURCE)
    image_processor = Compose([
        Resize(size=224, interpolation=Image.BICUBIC),
        CenterCrop(size=(224, 224)),
        lambda img: img.convert('RGB'),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    inputs=dict(pixel_values=image_processor(image_original).unsqueeze(0))
    with torch.no_grad():
        vision_outputs = _model.vision_model(**inputs)
        image_embeds = vision_outputs[1]
        image_embeds = _model.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # [1, 768]
        # adapt to SD 1.5's token dims
        image_embeds = image_embeds.unsqueeze(1)           # → [1, 1, 768]
        image_embeds = image_embeds.repeat(1, 77, 1)       # → [1, 77, 768]

    unconditional_embeds = torch.zeros_like(image_embeds)
    image_embeddings = torch.cat([unconditional_embeds, image_embeds]).to(device).to(dtype)

    # third option
    # model = CLIPVisionModelWithProjection.from_pretrained(CLIP_SOURCE)
    # processor = AutoProcessor.from_pretrained(CLIP_SOURCE)

    # inputs = processor(images=image_original, return_tensors="pt")

    # outputs = model(**inputs)
    # image_embeds = outputs.image_embeds
    # # adapt to SD 1.5's token dims
    # image_embeds = image_embeds.unsqueeze(1)           # → [1, 1, 768]
    # image_embeds = image_embeds.repeat(1, 77, 1)       # → [1, 77, 768]
    # # print(image_embeds.shape)

    # unconditional_embeds = torch.zeros_like(image_embeds)
    # image_embeddings = torch.cat([unconditional_embeds, image_embeds]).to(device).to(dtype)

    return image_embeddings


class PTD(nn.Module):
    def __init__(
        self, 
        dtype, 
        ref_img_path, 
        conditioning_scale: float = 1.0,
        prompt: str = "a photo of a japanese style living room",
        interpolate_scale: float = 1.0, # default: 1.0, ptd; 0.0, start from give image
        render_size: int = 512,
        contrast: float = 2., 
        add_noise: bool = False, 
        noise_value: float = 0.05,
        t_enc: int = 1000,
        t_dec: int = 100,
        blending_ratio_default: float = 1.0,
        direct_transfer_ratio: float = 0.4,
        decayed_transfer_ratio: float = 0.2,
        exponent: float = 0.5,
        guidance_scale: float = 7.5,
        async_ahead_steps: int = 0,
    ):
        """PTDiffusion implementation
        Args:
            render_size: size of the rendered image
        """

        super().__init__()

        self.batch_size = 1

        # set self configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype= dtype
        self.ref_img_path = ref_img_path
        self.conditioning_scale = conditioning_scale
        self.prompt = prompt
        # self.a_prompt = ", best quality, high quality, extremely detailed, good geometry, high-res photo"
        # self.n_prompt = "deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke, shading, lighting, lumination, shadow, text in image, watermarks"
        self.a_prompt = ""
        self.n_prompt = ""
        self.render_size = render_size
        self.contrast = contrast
        self.add_noise = add_noise
        self.noise_value = noise_value
        self.t_enc = t_enc
        self.t_dec = t_dec
        self.blending_ratio_default = blending_ratio_default
        self.direct_transfer_ratio = direct_transfer_ratio
        self.decayed_transfer_ratio = decayed_transfer_ratio
        self.exponent = exponent
        self.guidance_scale = guidance_scale
        self.interpolate_scale = interpolate_scale
        self.async_ahead_steps = async_ahead_steps

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
        self.pipe = pipe
        self.controlnet = controlnet
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler

        CONSOLE.print("PTDiffusion loaded!")

        # load the secret view's original image
        self.image_original = Image.open("data/fb5a96b1a2_original/DSC02791_original.png")

        # get the image embedding
        # image_embeddings = encode_image(self.image_original, self.dtype, self.device)
        # self.uncond, self.cond = image_embeddings.chunk(2)
        # self.text_embeddings = image_embeddings

        # get the text embedding
        text_embeddings = encode_prompt_with_a_prompt_and_n_prompt(self.batch_size, self.prompt, self.a_prompt, self.n_prompt, tokenizer, text_encoder, self.device, particle_num_vsd=1)
        self.uncond, self.cond = text_embeddings.chunk(2)
        self.text_embeddings = text_embeddings

        # self.ref_latent_path = 'latent_face1.jpg.pt'
        # self.ref_latent_path = './outputs/latent.pt'
        # self.ref_latent_path = './outputs/latent_face1.jpg_cond.pt'
        self.ref_latent_path = './outputs/latent_face1.jpg_uncond.pt'

        # start PTD DDIM inversion of the reference image
        print("Starting PTD DDIM inversion...")
        self.DDIM_inversion()

        # or load the inverted latents directly
        # self.ref_latent_init = torch.load(self.ref_latent_path).cuda().to(self.dtype)

        # copy the initial ref latent to prevent overwriting
        self.ref_latent = self.ref_latent_init.clone()

    def load_ref_img(
        self
    ):
        """Load reference image for image editing
        Returns:
            img_tensor: reference image tensor
        """
        img = Image.open(self.ref_img_path).convert('RGB').resize((self.render_size, self.render_size))
        img = torchvision.transforms.ColorJitter(contrast=(self.contrast, self.contrast))(img)
        img = np.array(img)
        if len(img.shape) == 2:
            print('Image is grayscale, stack the channels!')
            img = np.stack([img, img, img], axis=-1)
        img = (img.astype(np.float32) / 127.5) - 1.0           # -1 ~ 1
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()   # 1, 3, 512, 512
        if self.add_noise:
            noise = (torch.rand_like(img_tensor) - 0.5) / 0.5      # -1 ~ 1
            img_tensor = (1 - self.noise_value) * img_tensor + self.noise_value * noise

        return img_tensor.to(self.dtype)

    @torch.no_grad()
    def encode_ddim(
        self,
        ref_image_tensor: torch.FloatTensor,
    ):
        """
        Invert an image into the diffusion latent at timestep t_enc using a DDIM-like inversion.

        Args:
            image           : preprocessed image tensor (B x C x H x W), in [-1,1]

        Returns:
            latents: final noised latent at timestep t_enc
        """
        # 1. Prepare scheduler
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.t_enc, device=self.pipe.device)

        # 2. Encode image through VAE once
        # latents = self.pipe.vae.encode(image).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        h = self.pipe.vae.encoder(ref_image_tensor)
        moments = self.pipe.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(mean)
        latents = self.pipe.vae.config.scaling_factor * sample

        # 3. Iterative inversion
        encode_iterator = tqdm(scheduler.timesteps, desc='Encoding Image', total=self.t_enc)

        alphas_cumprod = scheduler.alphas_cumprod.to(self.pipe.device)
        for _, t in enumerate(encode_iterator):
            # t: 1000, 999, 998, …, 1
            noise_pred = self.pipe.unet(latents, 1000 - t, encoder_hidden_states=self.uncond).sample # 0 ~ 999 #TODO: this is wrong, we need to use uncond with "" as n_prompt

            alphas = alphas_cumprod[1000 - t - 1] if 1000 - t - 1 >= 0 else alphas_cumprod[1000 - t]
            alphas_next = alphas_cumprod[1000 - t]

            xt_weighted = (alphas_next / alphas).sqrt() * latents
            weighted_noise_pred = alphas_next.sqrt() * ((1 / alphas_next - 1).sqrt() - (1 / alphas - 1).sqrt()) * noise_pred
                                                                
            latents = xt_weighted + weighted_noise_pred

        return latents

    def DDIM_inversion(
        self
    ):
        """DDIM inversion of reference image for image editing
        Args:
            uncond: unconditional conditioning
        Returns:
            inverted latents
        """
        # DDIM inversion of ref image
        # should move this before the first stage, if before the second stage, the results would be different from normal results TODO: still debugging
        img_ref_tensor = self.load_ref_img()
        # reversion trajectory
        self.ref_latent_init = self.encode_ddim(img_ref_tensor)
        torch.save(self.ref_latent_init, self.ref_latent_path)

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
        # cpu FFT
        # ref_cpu = ref_latent.to(torch.float32).cpu()
        # x_cpu = x_dec.to(torch.float32).cpu()
        # ref_fft = torch.fft.fft2(ref_cpu, dim=(-2, -1)).to(x_dec.device) # must move the variables to cpu or FFT internal error, this happends for 4090, Ada and cuda-11.7, try to use 11.8.
        # x_fft = torch.fft.fft2(x_cpu,     dim=(-2, -1)).to(x_dec.device)

        # cuda FFT
        ref_fft = torch.fft.fft2(ref_latent.to(torch.float32), dim=(-2, -1)).to(x_dec.device) # must move the variables to cpu or FFT internal error, this happends for 4090, Ada and cuda-11.7, try to use 11.8.
        x_fft = torch.fft.fft2(x_dec.to(torch.float32),     dim=(-2, -1)).to(x_dec.device)

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
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
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
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            c_in = torch.cat([unconditional_conditioning, cond], dim=0)
            out = self.unet(x_in, t_in, encoder_hidden_states=c_in).sample
            e_uncond, e_cond = out.chunk(2)
            e_t = e_uncond + unconditional_guidance_scale * (e_cond - e_uncond)
        else:
            e_t = self.unet(x, t, encoder_hidden_states=cond).sample

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
        image: Float[Tensor, "BS 3 H W"] # [0, 1]
    ):
        """Edit an image using PTDiffusion
        Args:
            image: current rendered image
        Returns:
            edited image
        """
        # reset the ref latent to prevent reusing the results from last edition 
        self.ref_latent = self.ref_latent_init.clone()
        ############ add this to debug the results ###########
        # self.ref_latent = torch.randn_like(self.ref_latent)
        ############ add this to debug the results ###########
        # linear ref timesteps
        ref_update_steps = np.linspace(990, 0, self.t_dec)

        weights_decay_transfer = torch.linspace(self.blending_ratio_default, 0.0, int(self.decayed_transfer_ratio * self.t_dec)) ** self.exponent

        self.stage = "direct_transfer"

        # start from noise 
        # x_dec = torch.randn_like(self.ref_latent)
        # or start from the RGB image
        with torch.no_grad():
            latents = self.imgs_to_latent(image)
        noise = torch.randn_like(self.ref_latent)
        t_init = torch.tensor(int(ref_update_steps[0])).to(self.device)
        x_dec = self.scheduler.add_noise(latents, noise, t_init)

        loop = tqdm(range(self.t_dec))
        for i in loop:
            chosen_t = int(ref_update_steps[i])
            t = torch.tensor([chosen_t]).to(self.device)

            chosen_t_last = int(ref_update_steps[i + 1]) if i + 1 < self.t_dec else int(ref_update_steps[i])
            t_last = torch.tensor([chosen_t_last]).to(self.device)

            ts_tensor = torch.full((self.ref_latent.size(0),), chosen_t, device=self.ref_latent.device, dtype=torch.long) # 990 ~ 0
            
            ref_a_prev, ref_pred_x0, ref_e = self.p_sample_ddim(
                self.ref_latent, self.uncond, ts_tensor, t, t_last,
                return_all=True
            )
            dec_a_prev, dec_pred_x0, dec_e = self.p_sample_ddim(
                x_dec, self.cond, ts_tensor, t, t_last,
                unconditional_guidance_scale=self.guidance_scale,
                unconditional_conditioning=self.uncond,
                return_all=True
            )
            # reconstruct
            x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e

            # direct
            # self.ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e
            # async ahead reconstruction
            alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
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
            # x_dec = self.phase_substitute(self.ref_latent, x_prev, blending_ratio)
            # interpolation
            noisy_latents = self.scheduler.add_noise(latents, noise, t_last)
            phase_substituted_latents = self.phase_substitute(self.ref_latent, x_prev, blending_ratio)
            if self.stage == "refining":
                x_dec = self.interpolate_scale * phase_substituted_latents + (1 - self.interpolate_scale) * noisy_latents.to(self.dtype)
            else:
                x_dec = phase_substituted_latents

            loop.set_description(f"Stage: {self.stage}, Step: {i + 1}/{self.t_dec}, Blending Ratio: {blending_ratio:.4f}")

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
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        one_minus_alphas_sqrt = (1.0 - alphas_cumprod).sqrt()
        sigmas = torch.zeros_like(alphas_cumprod, device=device)

        # gather scalars for this step
        a_t      = alphas_cumprod[index]
        a_prev   = alphas_cumprod[last_index]
        sigma_t  = sigmas[index]
        sqrt_1ma = one_minus_alphas_sqrt[index]

        # reshape to match x
        a_t      = a_t.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        a_prev   = a_prev.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        sigma_t  = sigma_t.view(1, 1, 1, 1).expand(b, -1, -1, -1)
        sqrt_1ma = sqrt_1ma.view(1, 1, 1, 1).expand(b, -1, -1, -1)

        # predict epsilon
        if unconditional_guidance_scale != 1.0 and unconditional_conditioning is not None:
            x_in = torch.cat([x, x], dim=0)
            unet_cross_attention_kwargs = {'scale': 0}
            controlnet_output = self.controlnet(x_in.to(self.dtype), t, encoder_hidden_states=self.text_embeddings, controlnet_cond=controlnet_cond_input, conditioning_scale=self.conditioning_scale, guess_mode=False, return_dict=True)
            out = self.unet(x_in.to(torch.float16), t, encoder_hidden_states=self.text_embeddings, cross_attention_kwargs=unet_cross_attention_kwargs, down_block_additional_residuals=controlnet_output.down_block_res_samples, mid_block_additional_residual=controlnet_output.mid_block_res_sample).sample
            e_uncond, e_cond = out.chunk(2)
            e_t = e_uncond + unconditional_guidance_scale * (e_cond - e_uncond)
        else:
            # DDIM step for ref_latent
            e_t = self.unet(x, t, encoder_hidden_states=cond).sample

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
        image: Float[Tensor, "BS 3 H W"], # [0, 1]
        depth: Float[Tensor, "BS H W"] # [0, 1], meter
    ):
        """Edit an image using PTDiffusion
        Args:
            image: current rendered image
        Returns:
            edited image
        """
        # prepare depth condition input
        depth_tensor = self.get_depth_tensor(depth)
        controlnet_cond_input = torch.cat([depth_tensor] * 2)

        # reset the ref latent to prevent reusing the results from last edition 
        self.ref_latent = self.ref_latent_init.clone()

        ############ add this to debug the results ###########
        # self.ref_latent = torch.randn_like(self.ref_latent)
        ############ add this to debug the results ###########

        # linear ref timesteps
        ref_update_steps = np.linspace(990, 0, self.t_dec)

        weights_decay_transfer = torch.linspace(self.blending_ratio_default, 0.0, int(self.decayed_transfer_ratio * self.t_dec)) ** self.exponent

        self.stage = "direct_transfer"

        # start from noise 
        # x_dec = torch.randn_like(self.ref_latent)
        # or start from the RGB image
        with torch.no_grad():
            latents = self.imgs_to_latent(image)
        noise = torch.randn_like(self.ref_latent)
        t_init = torch.tensor(int(ref_update_steps[0])).to(self.device)
        x_dec = self.scheduler.add_noise(latents, noise, t_init)

        loop = tqdm(range(self.t_dec))
        for i in loop:
            chosen_t = int(ref_update_steps[i])
            t = torch.tensor([chosen_t]).to(self.device)

            chosen_t_last = int(ref_update_steps[i + 1]) if i + 1 < self.t_dec else int(ref_update_steps[i])
            t_last = torch.tensor([chosen_t_last]).to(self.device)

            ts_tensor = torch.full((self.ref_latent.size(0),), chosen_t, device=self.ref_latent.device, dtype=torch.long) # 990 ~ 0
            
            ref_a_prev, ref_pred_x0, ref_e = self.p_sample_ddim_depth(
                self.ref_latent, self.uncond, ts_tensor, t, t_last, controlnet_cond_input,
                return_all=True
            )
            dec_a_prev, dec_pred_x0, dec_e = self.p_sample_ddim_depth(
                x_dec, self.cond, ts_tensor, t, t_last, controlnet_cond_input,
                unconditional_guidance_scale=self.guidance_scale,
                unconditional_conditioning=self.uncond,
                return_all=True
            )
            # reconstruct
            x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e

            # direct
            # self.ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e
            # async ahead reconstruction
            alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
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
            # x_dec = self.phase_substitute(self.ref_latent, x_prev, blending_ratio)
            # interpolation
            noisy_latents = self.scheduler.add_noise(latents, noise, t_last)
            phase_substituted_latents = self.phase_substitute(self.ref_latent, x_prev, blending_ratio)
            if self.stage == "refining":
                x_dec = self.interpolate_scale * phase_substituted_latents + (1 - self.interpolate_scale) * noisy_latents.to(self.dtype)
            else:
                x_dec = phase_substituted_latents
            

            loop.set_description(f"Stage: {self.stage}, Step: {i + 1}/{self.t_dec}, Blending Ratio: {blending_ratio:.4f}")

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(x_dec)

        return decoded_img, depth_tensor
    
    def latents_to_img(self, latents):
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

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

        h = self.vae.encoder(imgs).to(self.device)
        moments = self.vae.quant_conv(h).to(self.device)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        std = torch.exp(0.5 * logvar).to(self.device)
        sample = mean + std * torch.randn_like(mean).to(self.device)
        latents = CONST_SCALE * sample

        return latents

        



        