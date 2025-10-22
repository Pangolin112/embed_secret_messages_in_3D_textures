import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import load_image

from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize

import random
from typing import Tuple
from PIL import Image, ImageDraw
from torchvision.utils import save_image
from torchvision import transforms


def generate_random_mask(
    size: int,
    num_dots: int = 20,
    dot_size_range: Tuple[int,int] = (10, 60),
    num_blocks: int = 10,
    block_size_range: Tuple[int,int] = (20, 100),
) -> Image.Image:
    # pure black background
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)

    # draw random dots
    for _ in range(num_dots):
        d = random.randint(*dot_size_range)
        r = d // 2
        cx = random.randint(0, size)
        cy = random.randint(0, size)

        # restrict the circle to be within the image bounds
        bbox = (
            max(cx-r, 0), max(cy-r, 0),
            min(cx+r, size), min(cy+r, size)
        )
        draw.ellipse(bbox, fill=255)

    # random rectangle blocks
    for _ in range(num_blocks):
        w = random.randint(*block_size_range)
        h = random.randint(*block_size_range)
        x0 = random.randint(0, size-w)
        y0 = random.randint(0, size-h)
        x1, y1 = x0 + w, y0 + h
        draw.rectangle((x0, y0, x1, y1), fill=255)

    return mask


image_original = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

# original unquantized FLUX FILL model loading
# pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")

# quantized FLUX FILL model
bfl_repo = "black-forest-labs/FLUX.1-Fill-dev"
transformer_quantize_path = "https://huggingface.co/boricuapab/flux1-fill-dev-fp8/blob/main/flux1-fill-dev-fp8.safetensors"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype_half = torch.float16

print("loading transformer")
# load with local files only
transformer = FluxTransformer2DModel.from_single_file("/home/qianru/.cache/huggingface/hub/models--boricuapab--flux1-fill-dev-fp8/snapshots/d14dd329cda6d3d1f7cb7e4c784d0ad11ea1fade/flux1-fill-dev-fp8.safetensors", torch_dtype=dtype_half, local_files_only=True).to(device)
# load with online files
# transformer = FluxTransformer2DModel.from_single_file(transformer_quantize_path, torch_dtype=dtype_half).to(device)

print("quantizing transformer")
quantize(transformer, weights=qfloat8)
freeze(transformer)

print("loading text_encoder")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype_half, local_files_only = True).to(device)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

print("loading pipe")
pipe = FluxFillPipeline.from_pretrained(bfl_repo, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype_half, local_files_only = True).to(device)
print("loading complete")

# image = pipe(
#     prompt="a white paper cup",
#     image=image,
#     mask_image=mask,
#     height=1632,
#     width=1232,
#     guidance_scale=30,
#     num_inference_steps=50,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]

render_size = 512 # [1232, 1632]
num_inference_steps = 50 #default 50
guidance_scale = 7.5 #30 #default 30
prompt = "a white paper cup" #default "a white paper cup"
num_dots = 0
num_blocks = 3
dot_size = 150 # default 150
rec_size = 50 # default 150

image_original = image_original.resize((render_size, render_size), resample=Image.Resampling.BILINEAR)
mask = mask.resize((render_size, render_size), resample=Image.Resampling.BILINEAR)

# mask = generate_random_mask(
#     size=render_size,
#     num_dots=num_dots,
#     dot_size_range=(dot_size, dot_size),
#     num_blocks=num_blocks,
#     block_size_range=(rec_size, rec_size),
# )

image = pipe(
    prompt=prompt,
    image=image_original,
    mask_image=mask,
    height=render_size,
    width=render_size,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
# image.save(f"outputs/FLUX_FILL/flux-fill-dev.png")

# save combined image
to_tensor = transforms.ToTensor()

image_original = to_tensor(image_original)
mask = to_tensor(mask)
image = to_tensor(image)

# repeat the mask 3 times to match the image channels
# mask = mask.repeat(3, 1, 1)

img_save = torch.cat([image_original, mask, image], dim=2)

# save_image(img_save, f"outputs/FLUX_FILL/guidance_{guidance_scale}_infer_{num_inference_steps}_prompt_{prompt}_num_dots_{num_dots}_num_blocks_{num_blocks}_dot_size_{dot_size}_rec_size_{rec_size}.png")
save_image(img_save, f"outputs/FLUX_FILL/guidance_{guidance_scale}_infer_{num_inference_steps}_prompt_{prompt}.png")
