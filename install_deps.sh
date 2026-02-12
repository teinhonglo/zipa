#!/bin/bash
source /home/slime-base/anaconda3/etc/profile.d/conda.sh
conda activate zipa_export

# PyTorch 2.4.0 + CUDA 12.4
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# k2
pip install k2==1.24.4.dev20241030+cuda12.4.torch2.4.0 -f https://k2-fsa.github.io/k2/cuda.html

# icefall
pip install git+https://github.com/k2-fsa/icefall.git

# Other dependencies
pip install onnx onnxruntime onnxconverter-common huggingface_hub sentencepiece lhotse
