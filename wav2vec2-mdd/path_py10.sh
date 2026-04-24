export KALDI_ROOT=/share/nas167/teinhonglo/kaldis/kaldi-20230627
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PYTHONNOUSERSITE=1
export WANDB_PROJECT=wav2vec2-mdd
export WANDB_MODE=offline
export WANDB_SILENT=true

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate wav2vec2-mdd-py10
