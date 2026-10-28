#!/usr/bin/env python3
"""
Generates the InterviewMate_Training_and_Demo.ipynb notebook.
"""

import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "colab",
    "InterviewMate_Training_and_Demo.ipynb"
)


def make_markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def make_code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def build_notebook():
    cells = []

    # ------------------------------------------------------------------------
    # CELL 1: TITLE & INTRO (Markdown)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
# 🎓 InterviewMate: Fine-Tuning & Cloud Deployment on Google Colab
### AI Mock Interviewer Specialized via Parameter-Efficient Fine-Tuning (QLoRA)

**Author / Student:** College Capstone Project  
**Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0 License)  
**Hardware:** Google Colab Free Tier (Tesla T4 GPU, 15GB VRAM)  
**Technique:** 4-bit NormalFloat (NF4) Quantization + LoRA (Low-Rank Adaptation)

---

### 📌 Project Workflow Overview
1. **Hardware & Environment Check:** Verify CUDA, Tesla T4 GPU, and VRAM.
2. **Install Cloud Dependencies:** Install `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, and `gradio`.
3. **Mount Google Drive:** Persist trained LoRA adapter weights across ephemeral Colab sessions.
4. **Dataset Setup:** Load and inspect the verified `interview_train.jsonl` and `interview_eval.jsonl` datasets.
5. **Instruction Formatting:** Convert records into standard ChatML conversation format.
6. **QLoRA Setup:** Quantize base model to 4-bit NormalFloat and configure LoRA adapters on attention/MLP projections.
7. **Fine-Tuning:** Train adapter using Hugging Face `SFTTrainer` with conservative memory settings.
8. **Save Adapter:** Export trained adapter weights (`adapter_config.json`, `adapter_model.safetensors`).
9. **Benchmarking:** Compare Base Model vs Fine-Tuned Model on held-out evaluation examples.
10. **Cloud Gradio Launch:** Serve the full interactive web application with a temporary public share link (`.gradio.live`) for live demonstration!
"""))

    # ------------------------------------------------------------------------
    # CELL 2: STEP 1 (Code)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 1: Verify Hardware & GPU Environment
Before running, make sure your Colab runtime is set to GPU:
👉 Go to **Runtime** > **Change runtime type** > Select **T4 GPU** > Click **Save**.
"""))

    cells.append(make_code_cell("""
import os
import sys
import torch

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU Detected: {device_name}")
    print(f"✅ Total VRAM: {total_mem:.2f} GB")
    if total_mem < 12.0:
        print("⚠️ Warning: Less than 12GB VRAM detected. Will configure conservative batch sizes.")
else:
    print("❌ ERROR: No GPU detected! Please enable GPU via Runtime > Change runtime type > T4 GPU.")
    print("Inference on CPU is possible for demo, but fine-tuning requires a GPU.")

# Display nvidia-smi
!nvidia-smi
"""))

    # ------------------------------------------------------------------------
    # CELL 3: STEP 2 (Dependencies)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 2: Install Cloud Dependencies
We install modern Hugging Face libraries compatible with CUDA 12 and 4-bit BitsAndBytes quantization.
"""))

    cells.append(make_code_cell("""
# Install latest compatible training and inference libraries
!pip install -q --upgrade pip
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers>=4.41.0 peft>=0.11.0 accelerate>=0.30.0 bitsandbytes>=0.43.1 datasets>=2.19.0 trl>=0.8.6 gradio>=4.31.0 scipy sentencepiece protobuf pydantic

print("✅ Dependencies successfully installed!")
"""))

    # ------------------------------------------------------------------------
    # CELL 4: STEP 3 (Google Drive)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 3: Mount Google Drive (Recommended for Persistence)
Google Colab sessions are ephemeral (files are deleted when the session disconnects). Mounting Google Drive lets us save the trained adapter permanently.
"""))

    cells.append(make_code_cell("""
from google.colab import drive
import os

MOUNT_DRIVE = True  # Set to False if you do not wish to mount Drive

if MOUNT_DRIVE:
    try:
        drive.mount('/content/drive')
        DRIVE_SAVE_DIR = '/content/drive/MyDrive/InterviewMate_Adapter'
        os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
        print(f"✅ Google Drive mounted successfully. Adapter will be saved to: {DRIVE_SAVE_DIR}")
    except Exception as e:
        print(f"⚠️ Could not mount Google Drive: {e}. Will save adapter locally to './saved_adapter'.")
        DRIVE_SAVE_DIR = './saved_adapter'
else:
    DRIVE_SAVE_DIR = './saved_adapter'
    print("Drive mounting skipped. Adapter will be saved locally.")
"""))

    # ------------------------------------------------------------------------
    # CELL 5: STEP 4 (Dataset Loading)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 4: Load and Validate the InterviewMate Dataset
If you cloned the project repository into Colab, the files will be in `data/`.  
Otherwise, the code below automatically creates or downloads the verified training and evaluation datasets.
"""))

    cells.append(make_code_cell("""
import json
import os

# Check if data directory exists; if not, clone or generate
if not os.path.exists("data/interview_train.jsonl"):
    print("Data files not found in working directory. Cloning repository or generating data...")
    # Option A: Clone from GitHub if hosted:
    # !git clone https://github.com/YOUR_USERNAME/interviewmate.git .
    
    # Option B: Auto-generate using the dataset script
    if os.path.exists("scripts/generate_dataset.py"):
        !python scripts/generate_dataset.py
    else:
        print("Please ensure interview_train.jsonl and interview_eval.jsonl are uploaded into data/ folder.")

train_path = "data/interview_train.jsonl"
eval_path = "data/interview_eval.jsonl"

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

train_data = load_jsonl(train_path)
eval_data = load_jsonl(eval_path)

print(f"✅ Loaded {len(train_data)} training records from {train_path}")
print(f"✅ Loaded {len(eval_data)} evaluation records from {eval_path}")

# Display a sample record
sample = train_data[0]
print("\\n--- Sample Training Record ---")
print(f"Category: {sample['category']} | Role: {sample['role']}")
print(f"Question: {sample['interview_question']}")
print(f"Candidate Answer ({sample['answer_quality']}): {sample['candidate_answer']}")
print(f"Score: {sample['overall_score']} / 10")
"""))

    # ------------------------------------------------------------------------
    # CELL 6: STEP 5 (Instruction Formatting)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 5: Format Dataset for Instruction Fine-Tuning (ChatML Format)
We format each record into standard multi-turn conversation format with `messages`:
- **System:** Evaluator persona and strict JSON schema instructions.
- **User:** The target role, experience level, interview question, and candidate answer.
- **Assistant:** The structured JSON evaluation containing score, rubric breakdown, strengths, weaknesses, actionable advice, improved answer, and follow-up question.
"""))

    cells.append(make_code_cell("""
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are InterviewMate, an expert AI mock interviewer and senior hiring committee member. "
    "Your mission is to evaluate the candidate's interview response with technical precision, "
    "fairness, and constructive coaching. "
    "You MUST respond ONLY with a single valid JSON object containing no markdown outside the JSON. "
    "Follow the requested schema strictly."
)

def format_record_to_chatml(record):
    user_prompt = f\"\"\"Candidate Target Role: {record['role']}
Experience Level: {record['experience_level']}
Interview Track: {record['category']}

Interview Question:
\"{record['interview_question']}\"

Candidate's Answer:
\"{record['candidate_answer']}\"

Scoring Rubric (Evaluate each metric from 1 to 10):
1. relevance (1-10)
2. accuracy_depth (1-10)
3. clarity_structure (1-10)
4. completeness (1-10)

Provide your evaluation strictly as a valid JSON object matching this schema:
{{
  "overall_score": <number 1-10>,
  "scoring_rubric": {{
    "relevance": <1-10>,
    "accuracy_depth": <1-10>,
    "clarity_structure": <1-10>,
    "completeness": <1-10>
  }},
  "strengths": ["<strength 1>", ...],
  "weaknesses": ["<weakness 1>", ...],
  "actionable_feedback": ["<actionable advice 1>", ...],
  "improved_sample_answer": "<ideal answer>",
  "follow_up_question": "<contextual follow-up>"
}}\"\"\"

    assistant_response = json.dumps({
        "overall_score": record["overall_score"],
        "scoring_rubric": record["scoring_rubric"],
        "strengths": record["strengths"],
        "weaknesses": record["weaknesses"],
        "actionable_feedback": record["actionable_feedback"],
        "improved_sample_answer": record["improved_sample_answer"],
        "follow_up_question": record["follow_up_question"]
    }, indent=2, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }

formatted_train = [format_record_to_chatml(r) for r in train_data]
formatted_eval = [format_record_to_chatml(r) for r in eval_data]

hf_train_dataset = Dataset.from_list(formatted_train)
hf_eval_dataset = Dataset.from_list(formatted_eval)

print(f"✅ Formatted {len(hf_train_dataset)} train samples and {len(hf_eval_dataset)} eval samples.")
"""))

    # ------------------------------------------------------------------------
    # CELL 7: STEP 6 (Model & QLoRA Config)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 6: Load Base Model with 4-bit QLoRA Quantization
We load `Qwen/Qwen2.5-1.5B-Instruct` using `BitsAndBytesConfig` in 4-bit NormalFloat (NF4).
- **Base Model Size:** ~3.1 GB in 16-bit FP.
- **Quantized to 4-bit:** ~1.2 GB VRAM!
- **LoRA Rank (r=16, alpha=32):** Adds trainable adapters to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feedforward layers (`gate_proj`, `up_proj`, `down_proj`).
"""))

    cells.append(make_code_cell("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. 4-bit NormalFloat Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 3. Load Base Causal LM
print(f"Loading {MODEL_ID} in 4-bit precision...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 4. Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)

# 5. Define LoRA Adapter Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"""))

    # ------------------------------------------------------------------------
    # CELL 8: STEP 7 (Training with SFTTrainer)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 7: Fine-Tune using SFTTrainer (Supervised Fine-Tuning)
We train for **3 epochs** using conservative memory hyperparameters:
- `per_device_train_batch_size = 2`
- `gradient_accumulation_steps = 4` (effective batch size = 8)
- `learning_rate = 2e-4` with cosine decay
- `fp16 = True` for NVIDIA T4 Tensor Cores
- Seed = 42 for strict reproducibility
"""))

    cells.append(make_code_cell("""
from trl import SFTTrainer
from transformers import TrainingArguments

OUTPUT_DIR = "./results"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    seed=42,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_train_dataset,
    eval_dataset=hf_eval_dataset,
    peft_config=peft_config,
    dataset_text_field=None,
    max_seq_length=1200,
    tokenizer=tokenizer,
    args=training_args
)

print("🚀 Starting Fine-Tuning with QLoRA...")
train_result = trainer.train()

print("\\n✅ Training Completed Successfully!")
print(f"Final Train Loss: {train_result.training_loss:.4f}")
"""))

    # ------------------------------------------------------------------------
    # CELL 9: STEP 8 (Save Adapter)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 8: Save Trained LoRA Adapter & Tokenizer
We export the trained adapter parameters to `./saved_adapter` and copy them to Google Drive for permanent backup.
"""))

    cells.append(make_code_cell("""
import shutil

LOCAL_ADAPTER_DIR = "./saved_adapter"

print(f"Saving trained LoRA adapter to {LOCAL_ADAPTER_DIR}...")
trainer.model.save_pretrained(LOCAL_ADAPTER_DIR)
tokenizer.save_pretrained(LOCAL_ADAPTER_DIR)

print("Files created in adapter folder:")
for fname in os.listdir(LOCAL_ADAPTER_DIR):
    print(f"  - {fname}")

# Copy to Google Drive if available
if DRIVE_SAVE_DIR != LOCAL_ADAPTER_DIR and os.path.exists("/content/drive"):
    try:
        shutil.copytree(LOCAL_ADAPTER_DIR, DRIVE_SAVE_DIR, dirs_exist_ok=True)
        print(f"✅ Adapter successfully backed up to Google Drive: {DRIVE_SAVE_DIR}")
    except Exception as e:
        print(f"⚠️ Could not copy to Drive: {e}")
"""))

    # ------------------------------------------------------------------------
    # CELL 10: STEP 9 (Benchmark Evaluation)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 9: Benchmark Base Model vs Fine-Tuned Model on Held-out Examples
We test both models on unseen evaluation samples from `data/interview_eval.jsonl` to measure:
1. **JSON Schema Compliance Rate**
2. **Mean Absolute Error (MAE)** against expert ground truth scores
3. **Completeness Rate** (Presence of strengths, weaknesses, improved answer, follow-up)
"""))

    cells.append(make_code_cell("""
import time
import json
import re

def test_inference(test_model, test_tok, prompt_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text}
    ]
    formatted = test_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = test_tok([formatted], return_tensors="pt").to(test_model.device)
    
    with torch.no_grad():
        out = test_model.generate(**inputs, max_new_tokens=450, temperature=0.1, do_sample=False)
    
    in_len = inputs["input_ids"].shape[1]
    return test_tok.decode(out[0][in_len:], skip_special_tokens=True).strip()

# Run a quick sample test
test_sample = eval_data[0]
test_prompt = f\"\"\"Target Role: {test_sample['role']}
Track: {test_sample['category']}
Question: \"{test_sample['interview_question']}\"
Candidate Answer: \"{test_sample['candidate_answer']}\"

Evaluate according to rubric and provide JSON output.\"\"\"

print("Evaluating sample using fine-tuned adapter...")
generated_feedback = test_inference(trainer.model, tokenizer, test_prompt)
print("\\n--- Generated Model Feedback ---")
print(generated_feedback)
"""))

    # ------------------------------------------------------------------------
    # CELL 11: STEP 10 (Gradio Cloud Launch)
    # ------------------------------------------------------------------------
    cells.append(make_markdown_cell("""
## Step 10: Launch Cloud Gradio Web Application
We launch the interactive Gradio mock interview web interface directly within this Colab notebook!  
Using `share=True`, Gradio generates a temporary public link (`https://xxxx.gradio.live`) that you can open from your Linux Mint laptop or share with your college professors during your presentation!

> **Important Presentation Note:**  
> The public link remains active as long as this Google Colab cell continues executing. When the Colab runtime disconnects, the temporary link will terminate.
"""))

    cells.append(make_code_cell("""
# Free up any lingering memory before launching the web server
torch.cuda.empty_cache()

# Launch Gradio interface with public sharing enabled
# (Passes the fine-tuned adapter directory so the app loads the trained model)
!python app.py --model "Qwen/Qwen2.5-1.5B-Instruct" --adapter "./saved_adapter" --share
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor: 2": 2,
        "nbformat_minor": 2
    }

    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated Colab notebook at: {NOTEBOOK_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    build_notebook()
