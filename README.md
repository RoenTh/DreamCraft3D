# Step 1: Set up a Python virtual environment
!python3 -m virtualenv venv
!source venv/bin/activate

# Step 2: Upgrade pip to the latest version
!python3 -m pip install --upgrade pip

# Step 3: Install PyTorch and Torchvision (choose one version)
# Option 1: Install torch1.12.1+cu113
!pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Option 2: Install the latest version with CUDA 11.8 support
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Step 4: Clone the DreamCraft3D repository and navigate into it
!git clone https://github.com/deepseek-ai/DreamCraft3D
%cd DreamCraft3D

# Step 5: Install additional dependencies
!pip install ninja
!pip install -r requirements.txt

# Step 6: Download necessary model checkpoints
%cd load/zero123
!wget -O stable_zero123.ckpt https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt

# Step 7: Prepare omnidata directory and download files
%cd ..
!mkdir -p omnidata
%cd omnidata
!sudo apt update
!sudo apt install python3-pip -y
!pip3 install gdown
!gdown 'https://drive.google.com/uc?id=1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
!gdown 'https://drive.google.com/uc?id=1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt

# Step 8: Navigate back to the main directory and install additional packages
%cd ../..
!pip install carvekit ninja lightning==2.0.0 omegaconf==2.3.0 jaxtyping typeguard diffusers transformers accelerate opencv-python tensorboard matplotlib imageio imageio[ffmpeg] trimesh bitsandbytes sentencepiece safetensors huggingface_hub libigl xatlas networkx pysdf PyMCubes wandb torchmetrics controlnet_aux einops kornia taming-transformers-rom1504 git+https://github.com/openai/CLIP.git open3d plotly mediapipe

# Install specific versions and additional packages
!pip install --upgrade setuptools
!pip install setuptools==69.5.1
!pip install git+https://github.com/ashawkey/envlight.git
!pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
!pip install git+https://github.com/NVlabs/nvdiffrast.git
!pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
!sudo apt-get install libgl1-mesa-glx -y
!pip install mmcv
!pip install peft

from huggingface_hub import login
hf_token = "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" #add your own token from HuggingFace
login(token=hf_token)

!python preprocess_image.py 000.png --recenter
%cd /DreamCraft3D
