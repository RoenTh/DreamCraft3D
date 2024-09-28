# Install the dependecies:

!python3 -m virtualenv venv
!. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
!python3 -m pip install --upgrade pip
# torch1.12.1+cu113
!pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!git clone https://github.com/RoenTh/DreamCraft3D.git
%cd DreamCraft3D
!pip install ninja
!pip install -r requirements.txt
%cd load/zero123
!wget -O stable_zero123.ckpt https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt
# Download stable_zero123.ckpt from https://huggingface.co/stabilityai/stable-zero123
%cd ..
!mkdir -p omnidata

# Change to the directory
%cd omnidata
!sudo apt update
!sudo apt install python3-pip -y
!pip3 install gdown
# Download the files using gdown
!gdown 'https://drive.google.com/uc?id=1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
!gdown 'https://drive.google.com/uc?id=1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
%cd ..
%cd ..
!pip install carvekit
!pip install ninja
!pip install lightning==2.0.0 omegaconf==2.3.0 jaxtyping typeguard diffusers transformers accelerate opencv-python tensorboard matplotlib imageio imageio[ffmpeg] trimesh bitsandbytes sentencepiece safetensors huggingface_hub libigl xatlas networkx pysdf PyMCubes wandb torchmetrics controlnet_aux
!pip install einops kornia taming-transformers-rom1504 git+https://github.com/openai/CLIP.git # zero123
!pip install open3d plotly # mesh visualization
!pip install --upgrade setuptools
!pip install mediapipe
!pip install setuptools==69.5.1
!pip install git+https://github.com/ashawkey/envlight.git
!pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
!pip install git+https://github.com/NVlabs/nvdiffrast.git
!pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
!sudo apt-get install libgl1-mesa-glx -y
!pip install mmcv


%cd /DreamCraft3D

# Install the diffusion repo in order to use DreamBooth

!git clone https://github.com/huggingface/diffusers
%cd diffusers
!pip install .

# Preprocess the reference image

!python preprocess_image_metric3D.py ref.png --recenter


# Create the input images for fine-tuning

!python threestudio/scripts/img_to_mv.py --image_path 'ref_rgba.png' --save_path 'diffusers/examples/dreambooth/ref' --prompt 'a photo of thin turbine-blade' --superres

# Train the dreambooth

!rm -r ref/.ipynb_checkpoints
!accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path="DeepFloyd/IF-I-XL-v1.0" \
    --instance_data_dir="ref" \
    --output_dir="blade/results" \
    --instance_prompt="a sks turbine-blade" \
    --resolution=64 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --scale_lr \
    --max_train_steps=1200 \
    --checkpointing_steps=600 \
    --pre_compute_text_embeddings \
    --tokenizer_max_length=77 \
    --text_encoder_use_attention_mask

# Lastly, train DreamCraft3D

!rm -rf .cache/*
!rm -rf ~/root/cache/*
!sudo rm -rf /var/cache/*
!sudo rm -rf /root/.cache/*
!sudo rm -rf /tmp/*
!rm -r ~/.local/share/Trash
!df
from huggingface_hub import login
hf_token = "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"
login(token=hf_token)
%cd /DreamCraft3D
prompt="a turbine-blade"
image_path="ref_rgba.png"
!python launch.py --config configs/dreamcraft3d-coarse-nerf.yaml --train system.prompt_processor.prompt="{prompt}" data.image_path="$image_path" system.guidance.lora_weights_path="diffusers/examples/dreambooth/blade/results" 
ckpt = f"outputs/dreamcraft3d-coarse-nerf/{prompt}@LAST/ckpts/last.ckpt"
!python launch.py --config configs/dreamcraft3d-coarse-neus.yaml --train system.prompt_processor.prompt="{prompt}" data.image_path="{image_path}" system.weights="{ckpt}" system.guidance.lora_weights_path="diffusers/examples/dreambooth/blade/results" 
ckpt = f"outputs/dreamcraft3d-coarse-neus/{prompt}@LAST/ckpts/last.ckpt"
!python launch.py --config configs/dreamcraft3d-geometry.yaml --train system.prompt_processor.prompt="{prompt}" data.image_path="{image_path}" system.geometry_convert_from="{ckpt}" system.guidance.lora_weights_path="diffusers/examples/dreambooth/blade/results"
ckpt = f"outputs/dreamcraft3d-geometry/{prompt}@LAST/ckpts/last.ckpt"
!python launch.py --config configs/dreamcraft3d-texture.yaml --train system.prompt_processor.prompt="{prompt}" data.image_path="{image_path}" system.geometry_convert_from="{ckpt}" 
