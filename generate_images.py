import os
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler # <<< NEW IMPORT: Import the scheduler class
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from peft import LoraConfig
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv() # Load environment variables

# --- Configuration (Adjust these parameters) ---
# Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, "output_lora_model")
OUTPUT_INFERENCE_DIR = os.path.join(PROJECT_ROOT, "generated_images")

# Inference Parameters
MODEL_NAME = "runwayml/stable-diffusion-v1-5" # Must match the model used for training
MIXED_PRECISION = "fp16" # Must match training precision for consistent loading

FORENSIC_TOKEN = "<forensic_details>" # Must match the token used for training

# Correct LoRA weights filename as saved by the training script
LORA_WEIGHTS_FILENAME = "adapter_model.safetensors"
LORA_WEIGHTS_PATH = os.path.join(OUTPUT_MODEL_DIR, LORA_WEIGHTS_FILENAME)

# Updated Prompts for Forensic Facial Images based on your training data
PROMPTS = [
    f"She is wearing earrings. She is {FORENSIC_TOKEN} attractive and has oval face.She is wearing earrings. She is attractive and has blond hair, and high cheekbones.She is wearing heavy makeup. She has oval face, and high cheekbones. She is young.The woman has high cheekbones. She wears earrings. She is attractive.This woman has oval face. She wears heavy makeup, and lipstick.She has oval face. She is attractive and is wearing earrings.This young person has blond hair. She has high cheekbones and is wearing heavy makeup. She is young.She is wearing heavy makeup, and earrings. She has high cheekbones, and oval face. She is attractive.The woman has blond hair, and high cheekbones and is wearing heavy makeup, and lipstick. She is attractive",
    f"a detailed image of {FORENSIC_TOKEN} a female face with long brown straight hair, heavy makeup, and a slight smile, as seen in a criminal database",
    # f"a forensic portrait of {FORENSIC_TOKEN} a young female, attractive face, wearing subtle earrings and lipstick, with a distinctive birthmark on her cheek, for a police file",
    f"This person has eyeglasses, black hair, and straight hair.This person has eyeglasses. He is chubby.The person has straight hair. He is young. He has no beard.This chubby person has big nose, and eyeglasses. This person is young and has straight hair, big nose, and black hair. He has straight hair.The man has big nose, and black hair. He is young.The man has eyeglasses, and big nose. He is chubby. He has no beard.This man has straight hair. He is young. He has no beard.The person has straight hair, black hair, and eyeglasses.",
    f"a high-quality image of {FORENSIC_TOKEN} a person with bright red wavy hair, a big nose, and bags under the eyes, neutral expression, evidence photo",
    f"a digitally reconstructed face of {FORENSIC_TOKEN} a young person with arched eyebrows and high cheekbones, showing features for forensic facial recognition",
    f"a grainy yet clear photo of {FORENSIC_TOKEN} a female face, dark straight hair, wearing a necklace, from surveillance footage analysis",
    f"a detailed forensic capture of {FORENSIC_TOKEN} an attractive young female with full lips and an intense gaze, for an investigation report",
    f"a side profile photo of {FORENSIC_TOKEN} a male with a prominent nose, wavy hair, and a strong jawline, suitable for forensic comparison"
]

NEGATIVE_PROMPT = "low quality, blurry, ugly, deformed, disfigured, poor lighting, cartoon, anime, illustration, painting, sketch, grayscale, bad anatomy, distorted features, extra limbs, extra fingers, poor composition"
NUM_INFERENCE_STEPS = 30 # Number of denoising steps for inference
GUIDANCE_SCALE = 7.5
NUM_SAMPLES_PER_PROMPT = 1 # Number of images to generate for each prompt
GENERATOR_SEED = 1200 # For reproducibility
# --- End Configuration ---

def plot_images(images, title="Generated Images"):
    if not images:
        print("No images to plot.")
        return

    num_images = len(images)
    cols = min(num_images, 4)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten axes for easy iteration
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')
    for i in range(num_images, len(axes)):
        axes[i].axis('off') # Hide unused subplots
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    os.makedirs(OUTPUT_INFERENCE_DIR, exist_ok=True)
    
    # Set torch dtype based on mixed precision
    torch_dtype = torch.float16 if MIXED_PRECISION == "fp16" else torch.float32

    pipeline = None # Initialize pipeline to None

    # --- Attempt to load directly from adapter_model.safetensors ---
    if os.path.exists(LORA_WEIGHTS_PATH) and os.path.getsize(LORA_WEIGHTS_PATH) > 0:
        try:
            print(f"Attempting to load LoRA weights from: {LORA_WEIGHTS_PATH}")
            
            # Load all base pipeline components explicitly
            tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", local_files_only=True)
            text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", local_files_only=True)
            vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae", local_files_only=True)
            unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet", local_files_only=True)
            scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler", local_files_only=True) # <<< FIX: Load scheduler here

            # --- CRITICAL: Add custom token and resize BEFORE loading LoRA weights ---
            num_added_tokens = tokenizer.add_tokens(FORENSIC_TOKEN)
            if num_added_tokens > 0:
                text_encoder.resize_token_embeddings(len(tokenizer))
                print(f"Added token '{FORENSIC_TOKEN}' and resized text encoder for base pipeline loading.")
            
            # Create the pipeline with all components
            pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler, # <<< FIX: Pass the loaded scheduler
                safety_checker=None, 
                feature_extractor=None, 
                requires_safety_checker=False 
            )

            # Load the LoRA weights using the full path
            pipeline.load_lora_weights(LORA_WEIGHTS_PATH)
            print("Successfully loaded LoRA weights from adapter_model.safetensors.")

        except Exception as e: # Catch any exception during direct loading
            print(f"Warning: Failed to load '{LORA_WEIGHTS_FILENAME}' directly ({e}). Trying full state checkpoints.")
            pipeline = None # Reset pipeline if loading failed
    else:
        print(f"'{LORA_WEIGHTS_FILENAME}' not found or is empty. Trying full state checkpoints instead.")
        pipeline = None

    # --- If direct loading failed or file not found, try loading from full state checkpoints ---
    if pipeline is None:
        checkpoints = os.listdir(OUTPUT_MODEL_DIR)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.replace("checkpoint-", "")))

        if not checkpoints:
            raise FileNotFoundError(f"No valid LoRA weights found, and no full state checkpoints in '{OUTPUT_MODEL_DIR}'. Please train your model first.")

        latest_checkpoint_dir = os.path.join(OUTPUT_MODEL_DIR, checkpoints[-1])
        print(f"Attempting to load from latest full state checkpoint: {latest_checkpoint_dir}")

        accelerator = Accelerator(mixed_precision=MIXED_PRECISION)
        
        # Load base components for the pipeline. These will be updated by accelerator.load_state
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", local_files_only=True)
        text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", local_files_only=True)
        vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae", local_files_only=True)
        unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet", local_files_only=True)
        scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler", local_files_only=True) # <<< FIX: Load scheduler here

        # --- CRITICAL: Add the custom token and resize the text encoder BEFORE loading the state ---
        num_added_tokens = tokenizer.add_tokens(FORENSIC_TOKEN)
        if num_added_tokens > 0:
            text_encoder.resize_token_embeddings(len(tokenizer))
            print(f"Added token '{FORENSIC_TOKEN}' and resized text encoder for checkpoint loading.")

        # Apply LoRA to the UNet (accelerator.load_state will then populate these adapters)
        lora_config = LoraConfig(
            r=64, # <<< ENSURE THESE MATCH YOUR TRAINING LoRA CONFIG >>>
            lora_alpha=32, # <<< ENSURE THESE MATCH YOUR TRAINING LoRA CONFIG >>>
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet.add_adapter(lora_config)

        # Prepare with accelerator for device placement before loading state
        unet, vae, text_encoder = accelerator.prepare(unet, vae, text_encoder)

        # Load the full state (this will load updated weights for unet and text_encoder)
        accelerator.load_state(latest_checkpoint_dir)
        print(f"Successfully loaded full state from {latest_checkpoint_dir}.")

        # Now, create the pipeline from the loaded (unwrapped) components
        pipeline = StableDiffusionPipeline(
            unet=accelerator.unwrap_model(unet),
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer, # Use the tokenizer with the added token
            scheduler=scheduler, # <<< FIX: Pass the loaded scheduler
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
    
    # Final check and move to device
    if pipeline is None:
        raise RuntimeError("Failed to load pipeline from any source. Check paths and training outputs.")
    
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pipeline moved to {'CUDA' if torch.cuda.is_available() else 'CPU'}.")


    # --- Generate Images ---
    all_generated_images = []
    
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(GENERATOR_SEED)

    for i, prompt in enumerate(PROMPTS):
        print(f"\nGenerating {NUM_SAMPLES_PER_PROMPT} images for prompt [{i+1}/{len(PROMPTS)}]: '{prompt}'")
        print(f"Negative prompt: '{NEGATIVE_PROMPT}'")

        with torch.no_grad(): # Inference does not require gradient calculation
            current_generated_images = pipeline(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                num_images_per_prompt=NUM_SAMPLES_PER_PROMPT,
                generator=generator,
            ).images
        
        all_generated_images.extend(current_generated_images)

        # Save images for the current prompt
        for j, img in enumerate(current_generated_images):
            img_path = os.path.join(OUTPUT_INFERENCE_DIR, f"generated_forensic_prompt_{i+1}_seed_{GENERATOR_SEED}_{j}.png")
            img.save(img_path)
            print(f"Saved {img_path}")

    print("\nAll image generation complete!")
    plot_images(all_generated_images, "All Generated Forensic Images")


if __name__ == "__main__":
    main()