#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Create a self-contained working directory for this experiment
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORK_DIR="$SCRIPT_DIR/work"
CACHE_DIR="$WORK_DIR/hf_cache"
DATA_DIR="$WORK_DIR/Kvasir-VQA-x1"
IMG_DIR="$DATA_DIR/images"
OUTPUT_DIR="$WORK_DIR/output"
export HF_HOME=$CACHE_DIR

mkdir -p "$WORK_DIR" "$CACHE_DIR" "$DATA_DIR" "$IMG_DIR" "$OUTPUT_DIR"

echo "--- 1. Preparing Dataset ---"
echo "Working directory: $WORK_DIR"

# This script block downloads the necessary image and QA data from Hugging Face
# and formats it into JSONL files as required by the ms-swift training framework.
python3 -c """
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Configuration from environment
CACHE_DIR = "$CACHE_DIR"
DATA_DIR = Path("$DATA_DIR")
IMG_DIR  = Path("$IMG_DIR")

IMAGE_DATASET = "SimulaMet/Kvasir-VQA-raw"
QA_DATASET = "SimulaMet/Kvasir-VQA-x1"

# a) Download and save unique images
print(f"⏬ Caching images from Hugging Face dataset: {IMAGE_DATASET}...")
try:
    image_ds = load_dataset(IMAGE_DATASET, split="raw", cache_dir=CACHE_DIR)
except Exception as e:
    print(f"Failed to load {IMAGE_DATASET}. Error: {e}")
    exit(1)

saved_images = set()
for item in tqdm(image_ds, desc="Saving unique images"):
    img_id = item['img_id']
    if img_id not in saved_images:
        img_path = IMG_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            item['image'].save(img_path)
        saved_images.add(img_id)
print(f"✅ Saved {len(saved_images)} unique images to {IMG_DIR}")

# b) Create JSONL files for training and validation
print(f"⏬ Creating JSONL files from Hugging Face dataset: {QA_DATASET}...")
def write_jsonl(split):
    out_path = DATA_DIR / f"Kvasir-VQA-x1-{split}.jsonl"
    ds = load_dataset(QA_DATASET, split=split, cache_dir=CACHE_DIR)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in tqdm(ds, desc=f"Processing {split} split"):
            rec = {
                "messages": [
                    {"role": "user", "content": f"<image>{r['question']}"},
                    {"role": "assistant", "content": r['answer']}
                ],
                "images": [str(IMG_DIR / f"{r['img_id']}.jpg")]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\\n')
    print(f"✅ Created {split} JSONL: {out_path}")

write_jsonl("train")
write_jsonl("test")
"""

echo "--- 2. Creating smaller validation set ---"
shuf -n 1000 "$DATA_DIR/Kvasir-VQA-x1-test.jsonl" > "$DATA_DIR/Kvasir-VQA-x1-test-1000.jsonl"
echo "✅ Created subsampled validation file."

echo "--- 3. Starting Fine-tuning ---"
# This command is adapted from the MediaEval-Medico-2025 sample notebook and Dockerfile.
# It uses ms-swift to fine-tune PaliGemma on the Kvasir-VQA-x1 dataset using LoRA.
swift sft \
    --dataset "$DATA_DIR/Kvasir-VQA-x1-train.jsonl" \
    --val_dataset "$DATA_DIR/Kvasir-VQA-x1-test-1000.jsonl" \
    --model_type "paligemma-3b-pt-224" \
    --max_length 512 \
    --sft_type lora \
    --torch_dtype float16 \
    --quantization_bit 4 \
    --bnb_4bit_compute_dtype float16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules "ALL" \
    --freeze_vit true \
    --gradient_checkpointing true \
    --load_best_model_at_end True \
    --metric_for_best_model eval/token_acc \
    --greater_is_better True \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 20 \
    --output_dir "$OUTPUT_DIR" \
    --push_to_hub false \
    --report_to "none" \
    --dataloader_num_workers 2 \
    --dataset_test_ratio 0 \
    --use_hf true \
    --cache_dir "$CACHE_DIR"

echo "--- ✅ Training complete ---"
echo "Model checkpoints and adapters are saved in $OUTPUT_DIR"
