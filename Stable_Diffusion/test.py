import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cuda"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("vocab.json", merges_file="merges.txt")
model_file = "inkpunk-diffusion-v1.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)


# TEXT TO IMAGE
prompt = "A cat with sunglasses, wearing comfy hat, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Optional: negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14
input_image = None  # No image for Text-to-Image
strength = 1  # Use 1 as a default value for Text-to-Image

# IMAGE TO IMAGE
# image_path = "cat.jpg"  # Path to input image
# prompt = "A cat with sunglasses, wearing comfy hat, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
# uncond_prompt = ""  # Optional: negative prompt
# do_cfg = True
# cfg_scale = 8  # min: 1, max: 14
# input_image = Image.open(image_path)
# strength = 0.8  # Strength to control how much transformation occurs

## SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cuda",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
# Convert NumPy array to PIL image
output_pil = Image.fromarray(output_image)

# Show the image
output_pil.show()

# Save the image (optional)
output_pil.save("output.png")
