# wav2vec2-mdd

### Dependencies
``` yaml
# conda environment  
conda create -n wav2vec2-mdd python==3.8.0
conda activate wav2vec2-mdd

# install requirements
pip install -r requirements.txt

# torch cuda version ( see https://pytorch.org/get-started/previous-versions) )  
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
#pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118


Assuming you've already installed HuggingFace transformers library, you need also to install the ctcdecode library
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .

# Configure the conda environment 
Modify the conda startup method in "path.sh" to your own path

# Reproducing experiments
./run_timit.sh --stage -1 --gpuid 0
./run_l2arctic.sh --stage -1 --gpuid 0
```
