# 🕵️ Forensic Image Generator

An advanced **AI-powered text-to-image generator** that uses fine-tuned **Stable Diffusion** to create forensic-style portraits based on detailed descriptions.

This project explores the application of generative AI in **forensic science and law enforcement**, providing a tool to visualize descriptions of individuals with high accuracy and detail.

---

## 🚀 Key Features
- **LoRA Fine-Tuning**: Efficiently trains a Stable Diffusion model using **Low-Rank Adaptation (LoRA)** on a custom dataset, preserving the base model's capabilities while specializing it for a forensic style.
- **Custom Token Conditioning**: Uses a dedicated token `<forensic_details>` to activate the model's knowledge of forensic image generation.
- **Streamlit Web App**: A user-friendly graphical interface for real-time image generation and parameter tuning.
- **Command-Line Inference**: A powerful script for generating multiple images from a predefined list of prompts, ideal for batch processing or testing.
- **Reproducible Environment**: Includes a `requirements.txt` file and clear instructions to set up the project locally or on cloud platforms like Google Colab.

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** with at least **8GB of VRAM** (12GB+ recommended for optimal performance).
- **Hugging Face account & access token**

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/014-Jayal/forensic_image_generator.git
cd forensic_image_generator
```

### 2. Set up a Python Virtual Environment
```bash
python -m venv .venv
# Activate the environment
# On Windows: .\.venv\Scriptsctivate
# On macOS/Linux: source ./.venv/bin/activate
```

### 3. Install PyTorch with CUDA support
Consult the official PyTorch installation page for the correct command. Example for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 5. Hugging Face Login
```bash
huggingface-cli login
```
Paste your token when prompted.

---

## Dataset Preparation
Create the following directories inside the `data` folder:
```text
data/
├── images/
└── captions.csv
```

- Place your training images (`.jpg` or `.png`) inside the `data/images` folder.
- Create a `captions.csv` file with two columns: **filename** and **caption**.

Example:
```csv
filename,caption
person1.jpg,<forensic_details> A middle-aged man with short brown hair and glasses
person2.png,<forensic_details> A young woman with long black hair and sharp jawline
```

---

## 💻 Usage

### 1. Fine-Tuning the Model
After setting up your dataset, run the training script to fine-tune the model with a LoRA adapter:
```bash
python train_lora.py
```
This will create the `output_lora_model/` folder with the trained weights.

### 2. Generating Images

#### a) Using the Streamlit App
For an interactive experience:
```bash
streamlit run app.py
```
This will start a local web server accessible in your browser (default: http://localhost:8501).

#### b) Using the Command Line
To generate images in batch from a predefined list of prompts:
```bash
python generate_images.py
```

---

## 📄 File Descriptions
- **train_lora.py** → The core training script. It loads the base model, adds a custom token, fine-tunes the UNet with LoRA, and saves checkpoints and final weights.
- **generate_images.py** → The inference script for generating images from a list of prompts. It loads the base model and applies the LoRA weights.
- **app.py** → The Streamlit web application providing a graphical user interface for real-time image generation.
- **requirements.txt** → Lists all Python libraries needed for the project.
- **data/** → Directory for storing the training dataset and captions.
- **output_lora_model/** → Directory where the fine-tuned model checkpoints and final LoRA weights are saved. (**Not included in the repository**)
- **generated_images/** → Directory where the output images from inference are stored. (**Not included in the repository**)

---

## 🤝 Contribution
Feel free to **open issues** or **submit pull requests**. All contributions are welcome!

---

## 👨‍💻 Authors
- **014-Jayal** – [GitHub](https://github.com/014-Jayal) | [LinkedIn](https://www.linkedin.com/in/jayal-shah04/)
- **Niheel Prajapati** – [GitHub](https://github.com/) | [LinkedIn](https://www.linkedin.com/in/niheel-prajapati/)

---

## ⚠️ Disclaimer
This project is for **educational and research purposes**.  
The generated images are **AI-generated** and should not be used as factual representations.  
All original models and code are used under their respective licenses.
