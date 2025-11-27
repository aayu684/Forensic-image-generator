import os
import torch
import math
import glob
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import shutil # For checkpoint cleanup

# Import from accelerate, transformers, diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig # For LoRA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Load environment variables (like Hugging Face Token)
from dotenv import load_dotenv
load_dotenv()

# --- Configuration (Adjust these parameters) ---
# Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")
CAPTIONS_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "captions.csv")
OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, "output_lora_model") # Base directory for final model and checkpoints

# Training Parameters
MODEL_NAME = "runwayml/stable-diffusion-v1-5" # Or "stabilityai/stable-diffusion-xl-base-1.0" for SDXL
RESOLUTION = 512 # Set to 1024 for SDXL

# --- OPTIMIZATION FOR SPEED (unchanged from previous fast version) ---
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = False
# --- END OPTIMIZATION FOR SPEED ---

LEARNING_RATE = 1e-4
LR_SCHEDULER = "cosine"
LR_WARMUP_STEPS = 0
NUM_TRAIN_EPOCHS = 100 # Adjust based on dataset size and desired convergence
MAX_TRAIN_STEPS = None # Set to None to train for NUM_TRAIN_EPOCHS (global default)
SEED = 42
MIXED_PRECISION = "fp16" # 'no', 'fp16', 'bf16' (bf16 for A100/V100 GPUs)

# --- CHECKPOINTING FOR RESUMPTION (UPDATED) ---
SAVE_EVERY_N_STEPS = 50 # Save a full state checkpoint every N steps (more robust for resuming)
CHECKPOINTS_TOTAL_LIMIT = 3 # Maximum number of full state checkpoints to keep
# --- END CHECKPOINTING FOR RESUMPTION ---

# LoRA Specific Parameters
RANK = 64
ALPHA = 32

# Forensic Specific Token
FORENSIC_TOKEN = "<forensic_details>"
PROMPT_TEMPLATE = f"a photo of {FORENSIC_TOKEN} {{}} , a high quality, detailed forensic image"
FILENAME_COLUMN = 'filename'
CAPTION_COLUMN = 'caption'
# --- End Configuration ---

# Set up logging
logger = get_logger(__name__)

# --- Dataset Class ---
class ForensicDataset(Dataset):
    def __init__(self, instance_data_root, captions_df, tokenizer, resolution=512, prompt_template=None,
                 filename_col='filename', caption_col='caption'):
        self.instance_data_root = instance_data_root
        self.captions_df = captions_df
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.prompt_template = prompt_template
        self.filename_col = filename_col
        self.caption_col = caption_col

        self.filename_to_caption = captions_df.set_index(filename_col)[caption_col].to_dict()

        all_image_files = glob.glob(os.path.join(instance_data_root, "*.jpg")) + \
                          glob.glob(os.path.join(instance_data_root, "*.png"))

        self.image_paths = [
            img_path for img_path in all_image_files
            if os.path.basename(img_path) in self.filename_to_caption
        ]

        if len(self.image_paths) != len(self.filename_to_caption):
            logger.warn(f"Warning: Found {len(self.image_paths)} images but {len(self.filename_to_caption)} captions in CSV.")
            logger.warn("This means some images don't have captions or vice-versa. Using only matched pairs.")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_filename = os.path.basename(image_path)

        image = Image.open(image_path).convert("RGB")
        image = self.image_transforms(image)

        caption = self.filename_to_caption.get(image_filename, "")

        if self.prompt_template:
            input_prompt = self.prompt_template.format(caption)
        else:
            input_prompt = caption

        input_ids = self.tokenizer(
            input_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return {"pixel_values": image, "input_ids": input_ids.squeeze()}
# --- End Dataset Class ---


def main():
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    logs_dir = os.path.join(OUTPUT_MODEL_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    set_seed(SEED)

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
        log_with="tensorboard",
        project_dir=logs_dir,
    )

    # Hugging Face Token (loaded from .env)
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set. Please add it to your .env file.")

    # Load scheduler, tokenizer and models.
    # Added local_files_only=True to ensure offline loading after first download
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler", local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME, subfolder="vae", local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_NAME, subfolder="unet", local_files_only=True
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if GRADIENT_CHECKPOINTING: # Now FALSE by default for speed
        unet.enable_gradient_checkpointing()

    # Add the special token to the tokenizer and text encoder
    num_added_tokens = tokenizer.add_tokens(FORENSIC_TOKEN)
    if num_added_tokens == 0:
        print(f"Warning: Token '{FORENSIC_TOKEN}' already exists in tokenizer. Consider using a truly unique token.")
    else:
        text_encoder.resize_token_embeddings(len(tokenizer))
        print(f"Added token '{FORENSIC_TOKEN}' to tokenizer and resized text encoder embeddings.")

    # Apply LoRA to the UNet
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    # Set up optimizer (only for LoRA parameters)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Load captions DataFrame
    if not os.path.exists(CAPTIONS_CSV_PATH):
        raise FileNotFoundError(f"Captions CSV not found at: {CAPTIONS_CSV_PATH}. Please ensure your dataset is correctly organized.")
    captions_df = pd.read_csv(CAPTIONS_CSV_PATH)
    print(f"Loaded captions from {CAPTIONS_CSV_PATH}. Found {len(captions_df)} entries.")


    train_dataset = ForensicDataset(
        instance_data_root=INSTANCE_DATA_DIR,
        captions_df=captions_df,
        tokenizer=tokenizer,
        resolution=RESOLUTION,
        prompt_template=PROMPT_TEMPLATE,
        filename_col=FILENAME_COLUMN,
        caption_col=CAPTION_COLUMN
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1
    ) # Use half CPU cores for num_workers

    # Calculate num_update_steps_per_epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)

    # Determine total training steps
    if MAX_TRAIN_STEPS is None:
        current_max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch
    else:
        current_max_train_steps = MAX_TRAIN_STEPS

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS * GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=current_max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    # >>> FIX HERE: Include vae and text_encoder in accelerator.prepare() <<<
    unet, optimizer, train_dataloader, lr_scheduler, vae, text_encoder = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, vae, text_encoder
    )

    # Remove the redundant .to() calls as accelerator.prepare() handles device placement and dtypes
    # vae.to(accelerator.device, dtype=torch.float16 if MIXED_PRECISION == "fp16" else torch.float32)
    # text_encoder.to(accelerator.device, dtype=torch.float16 if MIXED_PRECISION == "fp16" else torch.float32)

    global_step = 0
    first_epoch = 0

    # --- Resume from full state checkpoint logic (UPDATED) ---
    checkpoints = os.listdir(OUTPUT_MODEL_DIR)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace("checkpoint-", "")))

    if len(checkpoints) > 0:
        # Load the latest checkpoint
        path = os.path.join(OUTPUT_MODEL_DIR, checkpoints[-1])
        accelerator.load_state(path)
        global_step = int(checkpoints[-1].replace("checkpoint-", ""))
        
        # Calculate first_epoch based on loaded global_step
        # If num_update_steps_per_epoch is 0 (empty dataloader), avoid division by zero
        if num_update_steps_per_epoch > 0:
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            first_epoch = 0 # Or handle appropriately if dataset is empty

        print(f"Resuming training from checkpoint: {path}")
        print(f"Resumed at global_step: {global_step}, epoch: {first_epoch}")
    # --- END Resume from full state checkpoint logic ---

    print("\n***** Running Training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {NUM_TRAIN_EPOCHS}")
    print(f"  Instantaneous batch size per device = {TRAIN_BATCH_SIZE}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {TRAIN_BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Gradient Accumulation steps = {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Total optimization steps = {current_max_train_steps}")
    print(f"  Current starting epoch = {first_epoch}") # Display starting epoch
    print(f"  Current starting step = {global_step}") # Display starting step


    for epoch in range(first_epoch, NUM_TRAIN_EPOCHS):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps already completed if resuming within an epoch
            # This is important if global_step does not perfectly align with epoch start
            current_epoch_step = global_step % num_update_steps_per_epoch
            if accelerator.is_main_process and epoch == first_epoch and step < current_epoch_step:
                continue
            
            with accelerator.accumulate(unet):
                # Convert images to latent space
                # vae.dtype is now correctly set by accelerator.prepare()
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise to add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # global_step should increment *after* the optimizer step,
            # and logging should be conditioned on global_step.
            if accelerator.sync_gradients:
                global_step += 1 # Increment global_step here

                # Log and print only when global_step has actually incremented
                # and meets the logging frequency.
                if global_step % 10 == 0: # Log every 10 steps
                    accelerator.log({"loss": loss.item()}, step=global_step)
                    print(f"Epoch {epoch}/{NUM_TRAIN_EPOCHS}, Step {global_step}, Loss: {loss.item():.4f}")

                # --- Save full state checkpoint (UPDATED) ---
                if global_step % SAVE_EVERY_N_STEPS == 0:
                    output_checkpoint_dir = os.path.join(OUTPUT_MODEL_DIR, f"checkpoint-{global_step}")
                    accelerator.save_state(output_checkpoint_dir)
                    print(f"Saved full state checkpoint for step {global_step} to {output_checkpoint_dir}")

                    # Clean up old checkpoints
                    if CHECKPOINTS_TOTAL_LIMIT is not None:
                        checkpoints = os.listdir(OUTPUT_MODEL_DIR)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.replace("checkpoint-", "")))
                        if len(checkpoints) > CHECKPOINTS_TOTAL_LIMIT:
                            num_to_remove = len(checkpoints) - CHECKPOINTS_TOTAL_LIMIT
                            for i in range(num_to_remove):
                                old_checkpoint_dir = os.path.join(OUTPUT_MODEL_DIR, checkpoints[i])
                                if os.path.exists(old_checkpoint_dir):
                                    shutil.rmtree(old_checkpoint_dir)
                                    print(f"Removed old checkpoint: {old_checkpoint_dir}")
                # --- END Save full state checkpoint ---

            # Also check for max steps after global_step has potentially incremented
            if global_step >= current_max_train_steps:
                break # Exit the inner loop (dataloader)

        # After each epoch, save the final LoRA weights for easy deployment/testing
        # This is separate from the full state checkpoints
        accelerator.wait_for_everyone()
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_save_path = os.path.join(OUTPUT_MODEL_DIR, "adapter_model.safetensors")
        unwrapped_unet.save_pretrained(lora_save_path, safe_serialization=True)
        print(f"Saved LoRA adapter weights to {lora_save_path} after epoch {epoch}")


        # Check for max steps again after an epoch to ensure the outer loop also breaks
        if global_step >= current_max_train_steps:
            break # Exit the outer loop (epochs)

    accelerator.end()
    print("Training complete!")
    print(f"Final LoRA model saved to: {lora_save_path}")


if __name__ == "__main__":
    main()