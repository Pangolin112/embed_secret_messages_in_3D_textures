import torch
import torchvision
from torchvision.utils import save_image

from PIL import Image
import numpy as np

from tqdm import tqdm

from diffusers import DDIMScheduler


def load_ref_img(dtype, img_path, render_size=512, contrast=2., add_noise=False, noise_value=0.05):
    img = Image.open(img_path).convert('RGB').resize((render_size, render_size))
    img = torchvision.transforms.ColorJitter(contrast=(contrast, contrast))(img)
    img = np.array(img)
    if len(img.shape) == 2:
        print('Image is grayscale, stack the channels!')
        img = np.stack([img, img, img], axis=-1)
    img = (img.astype(np.float32) / 127.5) - 1.0           # -1 ~ 1
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()   # 1, 3, 512, 512
    if add_noise:
        noise = (torch.rand_like(img_tensor) - 0.5) / 0.5      # -1 ~ 1
        img_tensor = (1 - noise_value) * img_tensor + noise_value * noise
    return img_tensor.to(dtype)


def load_ref_img_grayscale(dtype, img_path, render_size=512, add_noise=False, noise_value=0.05):
    img = Image.open(img_path).resize((render_size, render_size))
    img = np.array(img)
    if len(img.shape) == 2:
        print('Image is grayscale, stack the channels!')
        img = np.stack([img, img, img], axis=-1)
    img = (img.astype(np.float32) / 127.5) - 1.0           # -1 ~ 1
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()   # 1, 3, 512, 512
    if add_noise:
        noise = (torch.rand_like(img_tensor) - 0.5) / 0.5      # -1 ~ 1
        img_tensor = (1 - noise_value) * img_tensor + noise_value * noise
    return img_tensor.to(dtype)


def load_texture_img(dtype, img_path, render_size=512, contrast=2., add_noise=False, noise_value=0.05):
    img = Image.open(img_path).convert('RGB').resize((render_size, render_size))
    img = torchvision.transforms.ColorJitter(contrast=(contrast, contrast))(img)
    img = np.array(img)
    if len(img.shape) == 2:
        print('Image is grayscale, stack the channels!')
        img = np.stack([img, img, img], axis=-1)
    img = (img.astype(np.float32) / 127.5) - 1.0           # -1 ~ 1
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()   # 1, 3, 512, 512
    if add_noise:
        noise = (torch.rand_like(img_tensor) - 0.5) / 0.5      # -1 ~ 1
        img_tensor = (1 - noise_value) * img_tensor + noise_value * noise
    return img_tensor.to(dtype)


@torch.no_grad()
def encode_ddim(
    pipeline,
    image: torch.FloatTensor,
    cond,
    t_enc: int
):
    """
    Invert an image into the diffusion latent at timestep t_enc using a DDIM-like inversion.

    Args:
        pipeline        : diffusers StableDiffusionPipeline (with .vae, .unet, .scheduler)
        image           : preprocessed image tensor (B x C x H x W), in [-1,1]
        cond            : model conditioning (e.g. text embeddings)
        t_enc           : number of diffusion steps to run (<= scheduler.num_inference_steps)

    Returns:
        latents: final noised latent at timestep t_enc
    """
    # 1. Prepare scheduler
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(t_enc, device=pipeline.device)

    # 2. Encode image through VAE once
    # latents = pipeline.vae.encode(image).latent_dist.sample() * pipeline.vae.config.scaling_factor
    h = pipeline.vae.encoder(image)
    moments = pipeline.vae.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    std = torch.exp(0.5 * logvar)
    sample = mean + std * torch.randn_like(mean)
    latents = pipeline.vae.config.scaling_factor * sample

    # 3. Iterative inversion
    encode_iterator = tqdm(scheduler.timesteps, desc='Encoding Image', total=t_enc)

    alphas_cumprod = scheduler.alphas_cumprod.to(pipeline.device)
    for idx, t in enumerate(encode_iterator):
        # t: 1000, 999, 998, …, 1
        noise_pred = pipeline.unet(latents, 1000 - t, encoder_hidden_states=cond).sample # 0 ~ 999

        alphas = alphas_cumprod[1000 - t - 1] if 1000 - t - 1 >= 0 else alphas_cumprod[1000 - t]
        alphas_next = alphas_cumprod[1000 - t]

        xt_weighted = (alphas_next / alphas).sqrt() * latents
        weighted_noise_pred = alphas_next.sqrt() * ((1 / alphas_next - 1).sqrt() - (1 / alphas - 1).sqrt()) * noise_pred
                                                            
        latents = xt_weighted + weighted_noise_pred

    return latents


def manual_fft2(x: torch.Tensor):
    """
    Compute the 2D FFT of a real input tensor x of shape [B, C, N, N]
    and return (magnitude, phase), each of shape [B, C, N, N].
    This uses only basic torch ops (sin, cos, matmul), so is fully differentiable.
    """
    B, C, H, W = x.shape
    assert H == W, "Only square spatial dimensions supported"
    N = H
    device = x.device
    # prepare DFT basis (N x N)
    n = torch.arange(N, device=device)
    k = n.view(N, 1)
    # exponent matrix: 2π * k * n / N
    exp_term = 2 * torch.pi * k * n / N
    cos_mat = torch.cos(exp_term).half()  # shape (N, N), symmetric
    sin_mat = torch.sin(exp_term).half()

    # flatten batch & channels -> (B*C, N, N)
    x_flat = x.reshape(-1, N, N)

    # First: DFT along width (last dim)
    #   real1[p, m, k] =  sum_n x[p, m, n] * cos(k,n)
    #   imag1[p, m, k] = -sum_n x[p, m, n] * sin(k,n)
    real1 = torch.matmul(x_flat, cos_mat)
    imag1 = -torch.matmul(x_flat, sin_mat)

    # Second: DFT along height (first spatial dim)
    #   real2[p, k, l] =  sum_m real1[p, m, l] * cos(k,m)
    #                   - sum_m imag1[p, m, l] * sin(k,m)
    #   imag2[p, k, l] =  sum_m imag1[p, m, l] * cos(k,m)
    #                   + sum_m real1[p, m, l] * sin(k,m)
    real2 = torch.matmul(cos_mat, real1) - torch.matmul(sin_mat, imag1)
    imag2 = torch.matmul(cos_mat, imag1) + torch.matmul(sin_mat, real1)

    # reshape back to [B, C, N, N]
    real2 = real2.view(B, C, N, N)
    imag2 = imag2.view(B, C, N, N)

    # magnitude and phase
    magnitude = torch.sqrt(real2**2 + imag2**2)
    phase     = torch.atan2(imag2, real2)

    return magnitude, phase


def manual_ifft2(magnitude: torch.Tensor, phase: torch.Tensor):
    """
    Inverse 2D FFT from magnitude & phase back to a real tensor of shape [B, C, N, N].
    This uses only torch.matmul, sin, cos, etc., so is fully differentiable.
    """
    # unpack and sanity-check
    B, C, N, N2 = magnitude.shape
    assert N == N2, "Only square spatial dims supported"
    device = magnitude.device

    # rebuild complex spectrum
    real_f = magnitude * torch.cos(phase)
    imag_f = magnitude * torch.sin(phase)

    # flatten batch & channels
    real_flat = real_f.view(-1, N, N)
    imag_flat = imag_f.view(-1, N, N)

    # build DFT basis (N x N)
    n = torch.arange(N, device=device)
    k = n.view(N, 1)
    exp_term = 2 * torch.pi * k * n / N
    cos_mat = torch.cos(exp_term).half()    # shape [N, N]
    sin_mat = torch.sin(exp_term).half()    # shape [N, N]

    # --- step 1: invert the DFT along the height (first spatial) dimension ---
    #    real1[m, l] = (1/N) * [ Σ_k ( cos[k,m] * real_f[k,l]
    #                             + sin[k,m] * imag_f[k,l] ) ]
    #   imag1[m, l] = (1/N) * [ Σ_k ( cos[k,m] * imag_f[k,l]
    #                             - sin[k,m] * real_f[k,l] ) ]
    real1 = (torch.matmul(cos_mat, real_flat) + torch.matmul(sin_mat, imag_flat)) / N
    imag1 = (torch.matmul(cos_mat, imag_flat) - torch.matmul(sin_mat, real_flat)) / N

    # --- step 2: invert the DFT along the width (second spatial) dimension ---
    #    x[m, n]  = (1/N) * [ Σ_l ( real1[m,l] * cos[l,n]
    #                            - imag1[m,l] * sin[l,n] ) ]
    x_flat = (torch.matmul(real1, cos_mat) - torch.matmul(imag1, sin_mat)) / N

    # reshape back to [B, C, N, N]
    return x_flat.view(B, C, N, N)


# phase substitute with manual_fft2 and manual_ifft2
# @torch.no_grad()
# def phase_substitute(
#     ref_latent: torch.FloatTensor,
#     x_dec: torch.FloatTensor,
#     alpha: float = 0.0
# ) -> torch.FloatTensor:
#     """
#     Substitute the phase of x_dec with ref_latent according to blending ratio alpha.
#     """
#     _, angle_ref = manual_fft2(ref_latent.to(torch.float16))
#     mag_x, angle_x = manual_fft2(x_dec.to(torch.float16))

#     # mix angles
#     mixed = angle_ref * (1 - alpha) + angle_x * alpha

#     # inverse FFT and restore to device
#     out = manual_ifft2(mag_x, mixed).to(x_dec.device)
#     return out.to(torch.float16)


# phase substitute with torch.fft.fft2
@torch.no_grad()
def phase_substitute(
    ref_latent: torch.FloatTensor,
    x_dec: torch.FloatTensor,
    alpha: float = 0.0
) -> torch.FloatTensor:
    """
    Substitute the phase of x_dec with ref_latent according to blending ratio alpha.
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

    return output.to(torch.float16)


@torch.no_grad()
def p_sample_ddim(
    x: torch.FloatTensor,
    cond: torch.FloatTensor,
    t: torch.LongTensor,
    index: int,
    last_index: int,
    scheduler,
    unet,
    parameterization: str = "eps",
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
    x = x.to(torch.float16)

    # pull precomputed arrays directly from scheduler
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
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
        out = unet(x_in, t_in, encoder_hidden_states=c_in).sample
        e_uncond, e_cond = out.chunk(2)
        e_t = e_uncond + unconditional_guidance_scale * (e_cond - e_uncond)
    else:
        e_t = unet(x, t, encoder_hidden_states=cond).sample

    # predict x0
    pred_x0 = (x - sqrt_1ma * e_t) / a_t.sqrt()

    if return_all:
        return a_prev, pred_x0, e_t

    # compute x_{t-1}
    dir_xt = torch.sqrt(1 - a_prev - sigma_t.pow(2)) * e_t
    noise = sigma_t * torch.randn_like(x)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    return x_prev.to(torch.float16)


@torch.no_grad()
def decode_with_phase_substitution(
    rgb_path,
    scheduler,
    unet,
    vae,
    ref_latent: torch.FloatTensor,
    cond:       torch.FloatTensor,
    t_dec:      int,
    guidance_scale:      float = 1.0,
    uncond_embedding:    torch.FloatTensor = None,
    direct_transfer_steps:  int = 55,
    decayed_transfer_steps: int = 0,
    blending_ratio:         float = 1.0,
    async_ahead_steps:      int = 0,
    exponent:               float = 0.5,
) -> torch.FloatTensor:
    """
    DDIM sampling with phase substitution, no use of `self`.
    scheduler: a DDIMScheduler with pre-set timesteps and device.
    unet: the StableDiffusion UNet model.
    """
    # prepare ranges
    # print("scheduler.timesteps: ", scheduler.timesteps)
    timesteps  = scheduler.timesteps[:t_dec]
    # time_range = list(reversed(timesteps))
    time_range = list(timesteps) # no need to reverse
    # print("time_range: ", time_range)
    direct_range = time_range[:direct_transfer_steps]
    decay_range  = time_range[direct_transfer_steps: direct_transfer_steps + decayed_transfer_steps]
    refine_range = time_range[direct_transfer_steps + decayed_transfer_steps:]

    direct_transfer_iterator = tqdm(direct_range, desc='Decoding image in the stage of direct phase transfer', total=direct_transfer_steps)                                 
    decayed_transfer_iterator = tqdm(decay_range, desc='Decoding image in the stage of decayed phase transfer', total=decayed_transfer_steps)            
    refining_iterator = tqdm(refine_range, desc='Decoding image in the refining stage', total=t_dec - direct_transfer_steps - decayed_transfer_steps)

    # start from noise 
    x_dec = torch.randn_like(ref_latent)

    # direct transfer
    for i, ts in enumerate(direct_transfer_iterator):
        ts_tensor = torch.full((ref_latent.size(0),), ts - 1, device=ref_latent.device, dtype=torch.long) # 990 ~ 0
        ref_a_prev, ref_pred_x0, ref_e = p_sample_ddim(
            ref_latent, uncond_embedding, ts_tensor, ts - 1,
            scheduler, unet, return_all=True
        )
        dec_a_prev, dec_pred_x0, dec_e = p_sample_ddim(
            x_dec, cond, ts_tensor, ts - 1,
            scheduler, unet,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond_embedding,
            return_all=True
        )
        # reconstruct
        x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e

        # direct
        ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e

        # phase transfer
        x_dec = phase_substitute(ref_latent, x_prev, blending_ratio)

    # save the intermediate result direct transfer
    ref_sample = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
    image_ref = vae.decode(ref_sample.to(torch.float16)).sample
    save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path}/ref_sample_direct_transfer.png')
    x_sample = 1 / vae.config.scaling_factor * x_dec.clone().detach()
    image_ = vae.decode(x_sample).sample.to(torch.float16)
    save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path}/x_sample_direct_transfer.png')

    # decayed transfer
    weights = torch.linspace(1, 0, decayed_transfer_steps) ** exponent
    for i, ts in enumerate(decayed_transfer_iterator):
        ts_tensor = torch.full((ref_latent.size(0),), ts - 1, device=ref_latent.device, dtype=torch.long)
        ref_a_prev, ref_pred_x0, ref_e = p_sample_ddim(
            ref_latent, uncond_embedding, ts_tensor, ts - 1,
            scheduler, unet, return_all=True
        )
        dec_a_prev, dec_pred_x0, dec_e = p_sample_ddim(
            x_dec, cond, ts_tensor, ts - 1,
            scheduler, unet,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond_embedding,
            return_all=True
        )
        # reconstruct
        x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e
        
        # direct
        ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e

        # phase transfer
        x_dec = phase_substitute(ref_latent, x_prev, float(weights[i]))

    # save the intermediate result decayed transfer
    ref_sample = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
    image_ref = vae.decode(ref_sample.to(torch.float16)).sample
    save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path}/ref_sample_decayed_transfer.png')
    x_sample = 1 / vae.config.scaling_factor * x_dec.clone().detach()
    image_ = vae.decode(x_sample).sample.to(torch.float16)
    save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path}/x_sample_decayed_transfer.png')

    # refining
    for i, ts in enumerate(refining_iterator):
        ts_tensor = torch.full((ref_latent.size(0),), ts - 1, device=ref_latent.device, dtype=torch.long)

        ref_a_prev, ref_pred_x0, ref_e = p_sample_ddim(
            ref_latent, uncond_embedding, ts_tensor, ts - 1,
            scheduler, unet, return_all=True
        )
        # direct
        ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e

        dec_a_prev, dec_pred_x0, dec_e = p_sample_ddim(
            x_dec, cond, ts_tensor, ts - 1,
            scheduler, unet,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond_embedding,
        return_all=True
        )
        # reconstruct
        x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e

        x_dec = x_prev.to(torch.float16)

    # save the intermediate result decayed transfer
    ref_sample = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
    image_ref = vae.decode(ref_sample.to(torch.float16)).sample
    save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path}/ref_sample_refine.png')
    x_sample = 1 / vae.config.scaling_factor * x_dec.clone().detach()
    image_ = vae.decode(x_sample).sample.to(torch.float16)
    save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path}/x_sample_refine.png')

    return x_dec


@torch.no_grad()
def p_sample_ddim_depth(
    x: torch.FloatTensor,
    cond: torch.FloatTensor,
    t: torch.LongTensor,
    index: int,
    last_index: int,
    scheduler,
    unet,
    controlnet,
    conditioning_scale,
    controlnet_cond_input,
    text_embeddings,
    parameterization: str = "eps",
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
    x = x.to(torch.float16)

    # pull precomputed arrays directly from scheduler
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
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
        unet_cross_attention_kwargs = {'scale': 0}
        controlnet_output = controlnet(x_in.to(torch.float16), t, encoder_hidden_states=text_embeddings, controlnet_cond=controlnet_cond_input, conditioning_scale=conditioning_scale, guess_mode=False, return_dict=True)
        out = unet(x_in.to(torch.float16), t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=unet_cross_attention_kwargs, down_block_additional_residuals=controlnet_output.down_block_res_samples, mid_block_additional_residual=controlnet_output.mid_block_res_sample).sample
        e_uncond, e_cond = out.chunk(2)
        e_t = e_uncond + unconditional_guidance_scale * (e_cond - e_uncond)
    else:
        e_t = unet(x, t, encoder_hidden_states=cond).sample

    # predict x0
    pred_x0 = (x - sqrt_1ma * e_t) / a_t.sqrt()

    if return_all:
        return a_prev, pred_x0, e_t

    # compute x_{t-1}
    dir_xt = torch.sqrt(1 - a_prev - sigma_t.pow(2)) * e_t
    noise = sigma_t * torch.randn_like(x)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

    return x_prev.to(torch.float16)


def decode_with_phase_substitution_depth(
    log_step,
    rgb_path,
    scheduler,
    unet,
    vae,
    controlnet,
    conditioning_scale,
    controlnet_cond_input,
    text_embeddings,
    ref_latent: torch.FloatTensor,
    cond:       torch.FloatTensor,
    t_dec:      int,
    guidance_scale:      float = 1.0,
    uncond_embedding:    torch.FloatTensor = None,
    direct_transfer_steps:  int = 55,
    decayed_transfer_steps: int = 0,
    blending_ratio:         float = 1.0,
    async_ahead_steps:      int = 0,
    exponent:               float = 0.5,
) -> torch.FloatTensor:
    """
    DDIM sampling with phase substitution, no use of `self`.
    scheduler: a DDIMScheduler with pre-set timesteps and device.
    unet: the StableDiffusion UNet model.
    """
    # prepare ranges
    # print("scheduler.timesteps: ", scheduler.timesteps)
    timesteps  = scheduler.timesteps[:t_dec]
    # time_range = list(reversed(timesteps))
    time_range = list(timesteps) # no need to reverse
    # print("time_range: ", time_range)
    direct_range = time_range[:direct_transfer_steps]
    decay_range  = time_range[direct_transfer_steps: direct_transfer_steps + decayed_transfer_steps]
    refine_range = time_range[direct_transfer_steps + decayed_transfer_steps:]

    direct_transfer_iterator = tqdm(direct_range, desc='Decoding image in the stage of direct phase transfer', total=direct_transfer_steps)                                 
    decayed_transfer_iterator = tqdm(decay_range, desc='Decoding image in the stage of decayed phase transfer', total=decayed_transfer_steps)            
    refining_iterator = tqdm(refine_range, desc='Decoding image in the refining stage', total=t_dec - direct_transfer_steps - decayed_transfer_steps)

    # start from noise 
    x_dec = torch.randn_like(ref_latent)
    # start from a good image latent
    # x_dec = #TODO: load a good image latent

    # direct transfer
    for i, ts in enumerate(direct_transfer_iterator):
        ts_tensor = torch.full((ref_latent.size(0),), ts - 1, device=ref_latent.device, dtype=torch.long) # 990 ~ 0
        
        ref_a_prev, ref_pred_x0, ref_e = p_sample_ddim(
            ref_latent, uncond_embedding, ts_tensor, ts - 1,
            scheduler, unet, return_all=True
        )
        # direct
        ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e

        dec_a_prev, dec_pred_x0, dec_e = p_sample_ddim_depth( # if not use depth version here, results would less align with the depth image
            x_dec, cond, ts_tensor, ts - 1,
            scheduler, unet,
            controlnet, conditioning_scale, controlnet_cond_input, text_embeddings,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond_embedding,
            return_all=True
        )
        # reconstruct
        x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e

        # phase transfer
        x_dec = phase_substitute(ref_latent, x_prev, blending_ratio)

        if i % log_step == 0:
            # save the intermediate result direct transfer
            ref_sample = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
            image_ref = vae.decode(ref_sample.to(torch.float16)).sample
            save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path}/ref_sample_{i}_step_{ts - 1}.png')
            x_sample = 1 / vae.config.scaling_factor * x_dec.clone().detach()
            image_ = vae.decode(x_sample).sample.to(torch.float16)
            save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path}/x_sample_{i}_step_{ts - 1}.png')

    # decayed transfer
    weights = torch.linspace(1, 0, decayed_transfer_steps) ** exponent
    for i, ts in enumerate(decayed_transfer_iterator):
        ts_tensor = torch.full((ref_latent.size(0),), ts - 1, device=ref_latent.device, dtype=torch.long)

        ref_a_prev, ref_pred_x0, ref_e = p_sample_ddim(
            ref_latent, uncond_embedding, ts_tensor, ts - 1,
            scheduler, unet, return_all=True
        )
        # direct
        ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e

        dec_a_prev, dec_pred_x0, dec_e = p_sample_ddim_depth(
            x_dec, cond, ts_tensor, ts - 1,
            scheduler, unet,
            controlnet, conditioning_scale, controlnet_cond_input, text_embeddings,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond_embedding,
            return_all=True
        )
        # reconstruct
        x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e
        
        # phase transfer
        x_dec = phase_substitute(ref_latent, x_prev, float(weights[i]))

        if i % log_step == 0:
            ref_sample = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
            image_ref = vae.decode(ref_sample.to(torch.float16)).sample
            save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path}/ref_sample_{direct_transfer_steps + i}_step_{ts - 1}.png')
            x_sample = 1 / vae.config.scaling_factor * x_dec.clone().detach()
            image_ = vae.decode(x_sample).sample.to(torch.float16)
            save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path}/x_sample_{direct_transfer_steps + i}_step_{ts - 1}.png')

    # refining
    for i, ts in enumerate(refining_iterator):
        ts_tensor = torch.full((ref_latent.size(0),), ts - 1, device=ref_latent.device, dtype=torch.long)

        ref_a_prev, ref_pred_x0, ref_e = p_sample_ddim(
            ref_latent, uncond_embedding, ts_tensor, ts - 1,
            scheduler, unet, return_all=True
        )
        # direct
        ref_latent = ref_a_prev.sqrt() * ref_pred_x0 + (1 - ref_a_prev).sqrt() * ref_e

        dec_a_prev, dec_pred_x0, dec_e = p_sample_ddim_depth(
            x_dec, cond, ts_tensor, ts - 1,
            scheduler, unet,
            controlnet, conditioning_scale, controlnet_cond_input, text_embeddings,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond_embedding,
            return_all=True
        )
        # reconstruct
        x_prev = dec_a_prev.sqrt() * dec_pred_x0 + (1 - dec_a_prev).sqrt() * dec_e

        x_dec = x_prev.to(torch.float16)

        if i % log_step == 0:
            ref_sample = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
            image_ref = vae.decode(ref_sample.to(torch.float16)).sample
            save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path}/ref_sample_{direct_transfer_steps + decayed_transfer_steps + i}_step_{ts - 1}.png')
            x_sample = 1 / vae.config.scaling_factor * x_dec.clone().detach()
            image_ = vae.decode(x_sample).sample.to(torch.float16)
            save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path}/x_sample_{direct_transfer_steps + decayed_transfer_steps + i}_step_{ts - 1}.png')

    # save the intermediate result decayed transfer
    ref_sample = 1 / vae.config.scaling_factor * ref_latent.clone().detach()
    image_ref = vae.decode(ref_sample.to(torch.float16)).sample
    save_image((image_ref / 2 + 0.5).clamp(0, 1), f'{rgb_path}/ref_sample_{t_dec}_step_{ts - 1}.png')
    x_sample = 1 / vae.config.scaling_factor * x_dec.clone().detach()
    image_ = vae.decode(x_sample).sample.to(torch.float16)
    save_image((image_ / 2 + 0.5).clamp(0, 1), f'{rgb_path}/x_sample_{t_dec}_step_{ts - 1}.png')

    return x_dec