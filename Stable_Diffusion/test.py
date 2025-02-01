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

tokenizer = CLIPTokenizer(r"C:\Users\sayed\Desktop\prsnl\LLMs\GitHub Content\Stable Diffusion\data\vocab.json", merges_file=r"C:\Users\sayed\Desktop\prsnl\LLMs\GitHub Content\Stable Diffusion\data\merges.txt")
model_file = r"C:\Users\sayed\Desktop\prsnl\LLMs\GitHub Content\Stable Diffusion\data\v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)



## TEXT TO IMAGE
prompt = "A man with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
# prompt = "A man stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14



## IMAGE TO IMAGE
# Comment to disable image to image
image_path = r"C:\Users\sayed\Desktop\prsnl\LLMs\GitHub Content\Stable Diffusion\images\me.JPG"
input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9



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