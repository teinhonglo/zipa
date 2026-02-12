
import sentencepiece as spm
import sys
import os

model_path = "ipa_simplified/unigram_127.model"
output_path = "ipa_simplified/tokens.txt"

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    sys.exit(1)

sp = spm.SentencePieceProcessor()
sp.Load(model_path)

with open(output_path, "w", encoding="utf-8") as f:
    for i in range(sp.GetPieceSize()):
        f.write(f"{sp.IdToPiece(i)} {i}\n")

print(f"Created {output_path} with {sp.GetPieceSize()} tokens")
