#!/bin/bash
source /home/slime-base/anaconda3/etc/profile.d/conda.sh
conda activate zipa_export

TOKENS="ipa_simplified/tokens.txt"

# Params for Small models
SMALL_LAYERS="2,2,3,4,3,2"
SMALL_FFN="512,768,1024,1536,1024,768"
SMALL_ENC="192,256,384,512,384,256"
SMALL_ENC_UNMASK="192,192,256,256,256,192"
SMALL_HEADS="4,4,4,8,4,4"
SMALL_CNN="31,31,15,15,15,31"
SMALL_QUERY=32
SMALL_VALUE=12
SMALL_POS_HEAD=4
SMALL_POS_DIM=48
SMALL_DS="1,2,4,8,4,2"

# Params for Large models
LARGE_LAYERS="4,3,4,5,4,4"
LARGE_FFN="768,768,1536,2048,1536,768"
LARGE_ENC="512,512,768,1024,768,512"
LARGE_ENC_UNMASK="192,192,256,320,256,192"
LARGE_HEADS="6,6,6,8,6,6"
# Assuming CNN kernel, query/value head dims etc. are same as small or inferred from train.py defaults if not specified.
# The README specified: --query-head-dim 64 --value-head-dim 48 --num-heads 6,6,6,8,6,6
LARGE_QUERY=64
LARGE_VALUE=48
# CNN kernel defaults in Zipformer2 are usually 31,31,15,15,15,31. I'll stick with that unless error.
LARGE_CNN="31,31,15,15,15,31" 
# Pos head/dim defaults
LARGE_POS_HEAD=4
LARGE_POS_DIM=48
LARGE_DS="1,2,4,8,4,2"

# Transducer Params
DEC_DIM=512 # Default for small
JOIN_DIM=512 # Default for small
LARGE_DEC_DIM=1024
LARGE_JOIN_DIM=1024

export_ctc() {
    NAME=$1
    MODEL_DIR="checkpoints/$NAME"
    LAYERS=$2
    FFN=$3
    ENC=$4
    ENC_UNMASK=$5
    HEADS=$6
    CNN=$7
    QUERY=$8
    VALUE=$9
    POS_HEAD=${10}
    POS_DIM=${11}
    DS=${12}

    echo "Exporting $NAME (CTC)..."
    /home/slime-base/anaconda3/envs/zipa_export/bin/python zipformer_crctc/export-onnx-ctc.py \
        --exp-dir $MODEL_DIR/exp \
        --tokens $TOKENS \
        --epoch 999 \
        --avg 1 \
        --use-averaged-model 0 \
        --num-encoder-layers "$LAYERS" \
        --feedforward-dim "$FFN" \
        --encoder-dim "$ENC" \
        --encoder-unmasked-dim "$ENC_UNMASK" \
        --num-heads "$HEADS" \
        --cnn-module-kernel "$CNN" \
        --query-head-dim $QUERY \
        --value-head-dim $VALUE \
        --pos-head-dim $POS_HEAD \
        --pos-dim $POS_DIM \
        --downsampling-factor "$DS" \
        --causal False \
        --use-transducer 0 \
        --use-ctc 1 \
        --fp16 True
}

export_transducer() {
    NAME=$1
    MODEL_DIR="checkpoints/$NAME"
    LAYERS=$2
    FFN=$3
    ENC=$4
    ENC_UNMASK=$5
    HEADS=$6
    CNN=$7
    QUERY=$8
    VALUE=$9
    POS_HEAD=${10}
    POS_DIM=${11}
    DS=${12}
    DEC_DIM=${13}
    JOIN_DIM=${14}

    echo "Exporting $NAME (Transducer)..."
    /home/slime-base/anaconda3/envs/zipa_export/bin/python zipformer_transducer/export-onnx.py \
        --exp-dir $MODEL_DIR/exp \
        --tokens $TOKENS \
        --epoch 999 \
        --avg 1 \
        --use-averaged-model 0 \
        --num-encoder-layers "$LAYERS" \
        --feedforward-dim "$FFN" \
        --encoder-dim "$ENC" \
        --encoder-unmasked-dim "$ENC_UNMASK" \
        --num-heads "$HEADS" \
        --cnn-module-kernel "$CNN" \
        --query-head-dim $QUERY \
        --value-head-dim $VALUE \
        --pos-head-dim $POS_HEAD \
        --pos-dim $POS_DIM \
        --downsampling-factor "$DS" \
        --causal False \
        --decoder-dim $DEC_DIM \
        --joiner-dim $JOIN_DIM \
        --fp16 True
}

# CSV to Arrays? Manual calls easier.

# Small CTC
export_ctc "zipa-cr-small-300k" "$SMALL_LAYERS" "$SMALL_FFN" "$SMALL_ENC" "$SMALL_ENC_UNMASK" "$SMALL_HEADS" "$SMALL_CNN" $SMALL_QUERY $SMALL_VALUE $SMALL_POS_HEAD $SMALL_POS_DIM "$SMALL_DS"
export_ctc "zipa-cr-small-500k" "$SMALL_LAYERS" "$SMALL_FFN" "$SMALL_ENC" "$SMALL_ENC_UNMASK" "$SMALL_HEADS" "$SMALL_CNN" $SMALL_QUERY $SMALL_VALUE $SMALL_POS_HEAD $SMALL_POS_DIM "$SMALL_DS"
export_ctc "zipa-cr-ns-small-700k" "$SMALL_LAYERS" "$SMALL_FFN" "$SMALL_ENC" "$SMALL_ENC_UNMASK" "$SMALL_HEADS" "$SMALL_CNN" $SMALL_QUERY $SMALL_VALUE $SMALL_POS_HEAD $SMALL_POS_DIM "$SMALL_DS"
export_ctc "zipa-cr-ns-small-nodiacritics-700k" "$SMALL_LAYERS" "$SMALL_FFN" "$SMALL_ENC" "$SMALL_ENC_UNMASK" "$SMALL_HEADS" "$SMALL_CNN" $SMALL_QUERY $SMALL_VALUE $SMALL_POS_HEAD $SMALL_POS_DIM "$SMALL_DS"

# Large CTC
export_ctc "zipa-cr-large-300k" "$LARGE_LAYERS" "$LARGE_FFN" "$LARGE_ENC" "$LARGE_ENC_UNMASK" "$LARGE_HEADS" "$LARGE_CNN" $LARGE_QUERY $LARGE_VALUE $LARGE_POS_HEAD $LARGE_POS_DIM "$LARGE_DS"
export_ctc "zipa-cr-large-500k" "$LARGE_LAYERS" "$LARGE_FFN" "$LARGE_ENC" "$LARGE_ENC_UNMASK" "$LARGE_HEADS" "$LARGE_CNN" $LARGE_QUERY $LARGE_VALUE $LARGE_POS_HEAD $LARGE_POS_DIM "$LARGE_DS"
export_ctc "zipa-cr-ns-large-800k" "$LARGE_LAYERS" "$LARGE_FFN" "$LARGE_ENC" "$LARGE_ENC_UNMASK" "$LARGE_HEADS" "$LARGE_CNN" $LARGE_QUERY $LARGE_VALUE $LARGE_POS_HEAD $LARGE_POS_DIM "$LARGE_DS"
export_ctc "zipa-cr-ns-large-nodiacritics-780k" "$LARGE_LAYERS" "$LARGE_FFN" "$LARGE_ENC" "$LARGE_ENC_UNMASK" "$LARGE_HEADS" "$LARGE_CNN" $LARGE_QUERY $LARGE_VALUE $LARGE_POS_HEAD $LARGE_POS_DIM "$LARGE_DS"

# Small Transducer
export_transducer "zipa-t-small-300k" "$SMALL_LAYERS" "$SMALL_FFN" "$SMALL_ENC" "$SMALL_ENC_UNMASK" "$SMALL_HEADS" "$SMALL_CNN" $SMALL_QUERY $SMALL_VALUE $SMALL_POS_HEAD $SMALL_POS_DIM "$SMALL_DS" $DEC_DIM $JOIN_DIM
export_transducer "zipa-t-small-500k" "$SMALL_LAYERS" "$SMALL_FFN" "$SMALL_ENC" "$SMALL_ENC_UNMASK" "$SMALL_HEADS" "$SMALL_CNN" $SMALL_QUERY $SMALL_VALUE $SMALL_POS_HEAD $SMALL_POS_DIM "$SMALL_DS" $DEC_DIM $JOIN_DIM

# Large Transducer
export_transducer "zipa-t-large-300k" "$LARGE_LAYERS" "$LARGE_FFN" "$LARGE_ENC" "$LARGE_ENC_UNMASK" "$LARGE_HEADS" "$LARGE_CNN" $LARGE_QUERY $LARGE_VALUE $LARGE_POS_HEAD $LARGE_POS_DIM "$LARGE_DS" $LARGE_DEC_DIM $LARGE_JOIN_DIM
export_transducer "zipa-t-large-500k" "$LARGE_LAYERS" "$LARGE_FFN" "$LARGE_ENC" "$LARGE_ENC_UNMASK" "$LARGE_HEADS" "$LARGE_CNN" $LARGE_QUERY $LARGE_VALUE $LARGE_POS_HEAD $LARGE_POS_DIM "$LARGE_DS" $LARGE_DEC_DIM $LARGE_JOIN_DIM

