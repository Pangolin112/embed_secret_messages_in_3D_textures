import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import AttnAddedKVProcessor, AttnAddedKVProcessor2_0, LoRAAttnAddedKVProcessor, LoRAAttnProcessor, SlicedAttnAddedKVProcessor


def extract_lora_diffusers(unet, device):
    ### ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833
    ### begin lora
    # Set correct lora layers
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor

        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        ).to(device)

    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # self.unet.requires_grad_(True)
    unet.requires_grad_(False)
    for param in unet_lora_layers.parameters():
        param.requires_grad_(True)
    # self.params_to_optimize = unet_lora_layers.parameters()
    ### end lora
    return unet, unet_lora_layers


def encode_prompt(batch_size, prompt, tokenizer, text_encoder, device):
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    return torch.cat([uncond_embeddings, text_embeddings])


def encode_prompt_with_a_prompt_and_n_prompt(batch_size, prompt, a_prompt, n_prompt, tokenizer, text_encoder, device, particle_num_vsd):
    text_input = tokenizer(
        [prompt + a_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [n_prompt] * batch_size, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    return torch.cat([uncond_embeddings[:particle_num_vsd], text_embeddings[:particle_num_vsd]])


def get_loss_weights(dtype_half, betas):
    num_train_timesteps = len(betas)
    betas = torch.tensor(betas) if not torch.is_tensor(betas) else betas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def loss_weight(t):
        # loss_weight_type: 1m_alphas_cumprod
        return sqrt_1m_alphas_cumprod[t] ** 2

    weights = []
    for i in range(num_train_timesteps):
        weights.append(loss_weight(i).to(dtype_half))

    return weights


def get_t_schedule_dreamtime(num_train_steps, num_steps, alphas_cumprod, work_dir):
    T = num_train_steps  # Total timesteps
    N = num_steps  # Number of sampling steps

    # Ensure alphas_cumprod is a NumPy array
    alphas_cumprod = np.array(alphas_cumprod)

    # Compute W_d(t)
    W_d = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)

    # Time steps from 1 to T
    t = np.arange(1, T + 1)

    # Parameters m and s
    m = T / 2
    s = 125

    # Compute W_p(t)
    W_p = np.exp(-((t - m) ** 2) / (2 * s ** 2))

    # Compute W(t)
    W = W_d * W_p

    # Normalize W(t)
    Z = np.sum(W)
    W_normalized = W / Z

    # Compute cumulative sum of W(t) from t' to T
    cum_W = np.cumsum(W_normalized[::-1])[::-1]

    # Compute t(i) for i in 1 to N
    t_list = []
    for i in range(1, N + 1):
        target = i / N
        # Find t' such that cumulative sum from t' to T is closest to target
        t_index = np.argmin(np.abs(cum_W - target))
        t_i = t[t_index]
        t_list.append(t_i)

    chosen_ts = np.array(t_list)

    plt.figure()
    plt.plot(chosen_ts, marker='.', markersize=1)
    plt.title('Time Schedule')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(os.path.join(work_dir, "time_schedule.png"))

    return chosen_ts


def get_noisy_latents(vae, scheduler, rgb, t, device):
    h = vae.encoder(rgb).to(device)
    moments = vae.quant_conv(h).to(device)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    std = torch.exp(0.5 * logvar).to(device)
    sample = mean + std * torch.randn_like(mean).to(device)
    latents = vae.config.scaling_factor * sample

    noise = torch.randn_like(latents).to(device)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    return latents, noise, noisy_latents


def get_noise_pred(text_embeddings, depth_tensor, noisy_latents, t, controlnet, unet, unet_phi, scheduler, guidance_scale, conditioning_scale, particle_num_vsd, unet_cross_attention_kwargs, cross_attention_kwargs):
    with torch.no_grad():
        text_embeddings_input = text_embeddings
        controlnet_cond_input = torch.cat([depth_tensor] * 2 * particle_num_vsd)
        latent_model_input = torch.cat([noisy_latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        controlnet_output = controlnet(latent_model_input, t, encoder_hidden_states=text_embeddings_input, controlnet_cond=controlnet_cond_input, conditioning_scale=conditioning_scale, guess_mode=False, return_dict=True)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_input, cross_attention_kwargs=unet_cross_attention_kwargs, down_block_additional_residuals=controlnet_output.down_block_res_samples, mid_block_additional_residual=controlnet_output.mid_block_res_sample).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # vsd
    with torch.no_grad():
        batch_size = noisy_latents.shape[0]
        # Process the control conditions through ControlNet
        controlnet_cond_input = torch.cat([depth_tensor] * particle_num_vsd)
        controlnet_output_phi = controlnet(noisy_latents, t, encoder_hidden_states=text_embeddings_input[batch_size:], controlnet_cond=controlnet_cond_input, conditioning_scale=conditioning_scale, guess_mode=False, return_dict=True)
        # guidance_scale := cfg_phi == 1.0
        noise_pred_phi = unet_phi(noisy_latents, t, encoder_hidden_states=text_embeddings_input[batch_size:], cross_attention_kwargs=cross_attention_kwargs, down_block_additional_residuals=controlnet_output_phi.down_block_res_samples, mid_block_additional_residual=controlnet_output_phi.mid_block_res_sample).sample

    grad = noise_pred - noise_pred_phi.detach()

    grad = torch.nan_to_num(grad)

    return grad, noise_pred.detach().clone(), noise_pred_phi.detach().clone()


def phi_vsd_grad_diffuser(dtype_half, unet_phi, controlnet, conditioning_scale, noisy_latents_phi, noise_phi, text_embeddings_phi, t_phi, cross_attention_kwargs, depth_tensor, particle_num_vsd):
    loss_fn = nn.MSELoss()
    # ref to https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py#L114

    batch_size = noisy_latents_phi.shape[0]

    # Process the control conditions through ControlNet
    noisy_latents_phi = noisy_latents_phi.to(dtype_half)
    controlnet_cond_input = torch.cat([depth_tensor] * particle_num_vsd)
    controlnet_output_phi = controlnet(noisy_latents_phi, t_phi, encoder_hidden_states=text_embeddings_phi[batch_size:], controlnet_cond=controlnet_cond_input, conditioning_scale=conditioning_scale, guess_mode=False, return_dict=True)
    noise_pred_phi = unet_phi(noisy_latents_phi, t_phi, encoder_hidden_states=text_embeddings_phi[batch_size:], cross_attention_kwargs=cross_attention_kwargs, down_block_additional_residuals=controlnet_output_phi.down_block_res_samples, mid_block_additional_residual=controlnet_output_phi.mid_block_res_sample).sample

    target = noise_phi
    loss = loss_fn(noise_pred_phi, target)

    return loss
