#!/bin/bash
set -euo pipefail

l2arctic_dir="/share/corpus/l2arctic_release_v4.0"
data_kaldi="data-kaldi/l2arctic"
exp_root="exp/l2arctic/zipa_pretrained"

# pretrained model config
model_path=""
model_repo_id="anyspeech/zipa-small-crctc-300k"
checkpoint_filename="zipa_small_crctc_300000_avg10.pth"
bpe_model="ipa_simplified/unigram_127.model"
iter=300000
avg=10

stage=0
stop_stage=3

. ./scripts/utils/parse_options.sh

GREEN='\033[0;32m'
NC='\033[0m'

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo -e "${GREEN}Stage 0: Prepare L2-ARCTIC metadata ...${NC}"
  python scripts/l2arctic/prepare_l2arctic.py \
    --l2arctic-dir "${l2arctic_dir}" \
    --output-dir "${data_kaldi}"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo -e "${GREEN}Stage 1: Resolve pretrained ZIPA checkpoint ...${NC}"

  if [ -z "${model_path}" ]; then
    mkdir -p checkpoints
    model_path=$(python - <<PY
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id="${model_repo_id}", filename="${checkpoint_filename}", local_dir="checkpoints"))
PY
)
  fi

  echo "Using model path: ${model_path}"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo -e "${GREEN}Stage 2: Run ZIPA inference on L2-ARCTIC test ...${NC}"

  decode_dir="${exp_root}/decode_test"
  mkdir -p "${decode_dir}"

  python scripts/l2arctic/zipa_l2arctic_infer.py \
    --wav-scp "${data_kaldi}/test/wav.scp" \
    --ref-phn "${data_kaldi}/test/transcript_phn_text" \
    --model-path "${model_path}" \
    --bpe-model "${bpe_model}" \
    --remove-sil \
    --output "${decode_dir}/recogs-test-iter-${iter}-avg-${avg}-use-averaged-model.txt"
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo -e "${GREEN}Stage 3: Evaluate PFER metrics ...${NC}"

  decode_dir="${exp_root}/decode_test"
  python scripts/evaluate.py "${exp_root}" "decode_test"
  cat "${decode_dir}/final_metrics.json"
fi

