import os
import torch
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_MODEL_DIR = "output_lora_model"
FORENSIC_TOKEN = "<forensic_details>"
LORA_WEIGHTS_FILENAME = "adapter_model.safetensors"
LORA_WEIGHTS_PATH = os.path.join(OUTPUT_MODEL_DIR, LORA_WEIGHTS_FILENAME)
MIXED_PRECISION = "fp16"
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
SEED = 1234

@st.cache_resource
def load_pipeline():
    from diffusers import StableDiffusionPipeline
    import torch

    try:
        # Load the complete pipeline directly
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Get tokenizer and text encoder from the pipeline
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        
        # Add custom token for forensic details
        tokenizer.add_tokens(FORENSIC_TOKEN)
        text_encoder.resize_token_embeddings(len(tokenizer))
        
        # Load LoRA weights if they exist
        if os.path.exists(LORA_WEIGHTS_PATH):
            pipe.load_lora_weights(LORA_WEIGHTS_PATH)
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        
        return pipe
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        raise

# --- Streamlit UI --- 
import streamlit as st

# Constants
FORENSIC_TOKEN = "[FORENSIC]"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .section-header {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .tip-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .tip-box h4, .tip-box li {
        color: #000 !important;
    }
    
    .settings-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .settings-box h4, .settings-box li, .settings-box strong {
        color: #000 !important;
    }
    
    .section-header h3 {
        color: #000 !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎭 Forensic Face Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a detailed forensic description to generate a realistic face image using advanced AI technology</p>', unsafe_allow_html=True)

# Create main layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    # Main input section
    st.markdown('<div class="section-header"><h3>📝 Forensic Description</h3></div>', unsafe_allow_html=True)
    
    prompt_input = st.text_area(
        "Detailed Description",
        value=f"a forensic portrait of {FORENSIC_TOKEN} a young female with high cheekbones, wearing lipstick and earrings",
        height=120,
        help="Be as detailed as possible for better results. Include features like age, gender, facial structure, and accessories."
    )
    
    # Negative prompt in an expander for cleaner look
    with st.expander("🚫 Negative Prompt (Advanced)", expanded=False):
        negative_prompt = st.text_input(
            "What to avoid in the generation",
            value="blurry, low quality, cartoon, sketch",
            help="Specify what you want to exclude from the generated image"
        )

with col2:
    # Tips section
    st.markdown("""
    <div class="tip-box">
        <h4 style="color: black;">💡 Pro Tips</h4>
        <ul>
            <li style="color: black;">Include specific facial features and proportions</li>
            <li style="color: black;">Mention age range and gender for better accuracy</li>
            <li style="color: black;">Add details about hair, eyes, and skin tone</li>
            <li style="color: black;">Describe clothing and accessories if relevant</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Advanced settings section
st.markdown('<div class="section-header"><h3>⚙️ Generation Settings</h3></div>', unsafe_allow_html=True)

# Create two columns for sliders
slider_col1, slider_col2 = st.columns(2)

with slider_col1:
    st.markdown("**🔄 Inference Steps**")
    num_steps = st.slider(
        "Steps",
        min_value=10,
        max_value=100,
        value=NUM_INFERENCE_STEPS,
        help="Higher values produce more detailed results but take longer to generate"
    )
    st.caption(f"Current: {num_steps} steps")

with slider_col2:
    st.markdown("**🎯 Guidance Scale**")
    guidance = st.slider(
        "Guidance",
        min_value=1.0,
        max_value=15.0,
        value=GUIDANCE_SCALE,
        step=0.1,
        help="Controls how closely the AI follows your prompt. Higher values mean stricter adherence"
    )
    st.caption(f"Current: {guidance:.1f}")

# Settings guide
st.markdown("""
<div class="settings-box">
    <h4 style="color: black;">⚙️ Settings Guide</h4>
    <ul>
        <li style="color: black;"><strong style="color: black;">Steps:</strong> 20-50 for quick results, 50-100 for high quality</li>
        <li style="color: black;"><strong style="color: black;">Guidance:</strong> 5-10 for creative freedom, 10-15 for precise control</li>
        <li style="color: black;"><strong style="color: black;">Negative Prompts:</strong> Use to avoid unwanted elements like blur or cartoonish features</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Generate button with custom styling
st.markdown("---")
generate_button = st.button("🚀 Generate Forensic Portrait", type="primary")

# Display current settings summary
if st.checkbox("📊 Show Generation Summary", value=False):
    st.info(f"""
    **Current Settings:**
    - **Prompt:** {prompt_input[:100]}{'...' if len(prompt_input) > 100 else ''}
    - **Negative Prompt:** {negative_prompt}
    - **Inference Steps:** {num_steps}
    - **Guidance Scale:** {guidance}
    """)

# Add some spacing and footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>🔬 Advanced AI-powered forensic portrait generation • Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)

# Handle generation (you can add your generation logic here)
if generate_button:
    with st.spinner("🎨 Generating forensic portrait..."):
        # Your generation logic here
        st.success("✅ Generation parameters configured successfully!")
        st.balloons()
        
        # Display the parameters that would be used
        st.json({
            "prompt": prompt_input,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance
        })
if generate_button and prompt_input:
    pipe = load_pipeline()
    with torch.no_grad():
        generator = torch.Generator(device=pipe.device).manual_seed(SEED)
        result = pipe(
            prompt=prompt_input,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            num_images_per_prompt=1,
            generator=generator,
        )
        image = result.images[0]
        st.image(image, caption="Generated Forensic Face", use_container_width=True)
        output_path = os.path.join("generated_images", "streamlit_output.png")
        os.makedirs("generated_images", exist_ok=True)
        image.save(output_path)
        st.success(f"Image saved to {output_path}")