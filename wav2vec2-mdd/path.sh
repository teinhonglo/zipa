export KALDI_ROOT=/share/nas167/teinhonglo/kaldis/kaldi-20230627
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
export PYTHONPATH=$PWD
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PYTHONNOUSERSITE=1
export WANDB_PROJECT=wav2vec2-mdd
export WANDB_MODE=offline
export WANDB_SILENT=true

CUDA_116_DIR=/usr/local/cuda-11.6

if [ -d $CUDA_116_DIR ]; then
    export PATH=$CUDA_116_DIR/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_116_DIR/lib64:$LD_LIBRARY_PATH
fi

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate wav2vec2-mdd
