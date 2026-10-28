"""
InterviewMate Model Loader
==========================
Handles loading the pretrained open-weight LLM (Qwen2.5-1.5B-Instruct / Qwen3-1.7B)
and its fine-tuned LoRA / QLoRA adapters for inference.

Features:
- 4-bit QLoRA quantization via BitsAndBytesConfig on Colab GPUs.
- Automatic device fallback (CUDA GPU -> CPU).
- Support for both Base Model mode (before fine-tuning) and LoRA Adapter mode.
- Mock fallback mode for local testing without downloading multi-gigabyte models.
"""

import os
import sys
from typing import Optional, Tuple, Any, Dict

# Optional imports for environments where PyTorch/Transformers may not be installed locally
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    HF_AVAILABLE = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    PeftModel = None
    HF_AVAILABLE = False


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_LIGHT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def check_gpu_status() -> Dict[str, Any]:
    """Inspects available hardware and memory for logging and diagnostics."""
    status = {
        "hf_available": HF_AVAILABLE,
        "cuda_available": False,
        "device_count": 0,
        "device_name": "CPU Only",
        "total_memory_gb": 0.0,
        "free_memory_gb": 0.0,
        "recommended_quantization": "None (CPU)"
    }
    if HF_AVAILABLE and torch.cuda.is_available():
        status["cuda_available"] = True
        status["device_count"] = torch.cuda.device_count()
        status["device_name"] = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024 ** 3)
        free_mem = total_mem - reserved_mem
        status["total_memory_gb"] = round(total_mem, 2)
        status["free_memory_gb"] = round(free_mem, 2)
        status["recommended_quantization"] = "4-bit (NF4 QLoRA)" if total_mem < 20 else "8-bit / 16-bit"

    return status


def get_quantization_config(use_4bit: bool = True) -> Optional[Any]:
    """Returns BitsAndBytesConfig for 4-bit NormalFloat quantization."""
    if not HF_AVAILABLE or not torch.cuda.is_available() or not use_4bit:
        return None

    try:
        compute_dtype = torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
    except Exception as e:
        print(f"[ModelLoader] BitsAndBytesConfig initialization warning: {e}. Falling back to standard precision.")
        return None


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True
) -> Tuple[Any, Any, bool]:
    """
    Loads base model and tokenizer, optionally attaching a fine-tuned LoRA adapter.

    Returns:
        (model, tokenizer, is_adapter_active)
    """
    if not HF_AVAILABLE:
        print("[ModelLoader] Warning: Transformers/PyTorch not installed. Using MockModel for UI/local verification.")
        return MockModel(adapter_path=adapter_path), MockTokenizer(), bool(adapter_path)

    print(f"[ModelLoader] Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quantization_config(use_4bit=load_in_4bit)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[ModelLoader] Loading base model '{model_name}' (device_map={device_map}, 4bit={quant_config is not None})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch_dtype if quant_config is None else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    is_adapter_active = False
    if adapter_path and os.path.exists(adapter_path):
        print(f"[ModelLoader] Found LoRA adapter at '{adapter_path}'. Attaching to base model...")
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            is_adapter_active = True
            print("[ModelLoader] Successfully attached fine-tuned LoRA adapter!")
        except Exception as e:
            print(f"[ModelLoader] Error attaching adapter: {e}. Continuing with base model.")
            is_adapter_active = False
    else:
        if adapter_path:
            print(f"[ModelLoader] Adapter path '{adapter_path}' does not exist. Using base model.")
        else:
            print("[ModelLoader] No adapter specified. Running in Base Model mode.")

    return model, tokenizer, is_adapter_active


def generate_model_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9
) -> str:
    """
    Generates a response from the model given a prompt.
    Applies ChatML chat template for Qwen if supported.
    """
    if isinstance(model, MockModel):
        return model.generate_mock_feedback(prompt)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback if chat template is not configured
        formatted_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

    inputs = tokenizer([formatted_prompt], return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Slice off input tokens to retrieve newly generated response
    input_length = inputs["input_ids"].shape[1]
    response_tokens = output_ids[0][input_length:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    return response_text


# ==============================================================================
# MOCK / FALLBACK ENGINE FOR LOCAL UI DEVELOPMENT
# (Allows running and testing the Gradio UI on Linux Mint laptop without GPU/torch)
# ==============================================================================

class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def decode(self, *args, **kwargs):
        return ""


class MockModel:
    def __init__(self, adapter_path: Optional[str] = None):
        self.device = "cpu"
        self.adapter_path = adapter_path

    def generate_mock_feedback(self, prompt: str) -> str:
        """Produces realistic structured JSON response for local testing."""
        import json
        is_weak = any(w in prompt.lower() for w in ["don't know", "fast", "terrible", "hackers", "delete"])
        score = 4 if is_weak else 8

        mock_payload = {
            "overall_score": score,
            "scoring_rubric": {
                "relevance": score,
                "accuracy_depth": max(1, score - 1),
                "clarity_structure": score,
                "completeness": score
            },
            "strengths": [
                "Clear attempt to answer the core technical question.",
                "Demonstrated awareness of the underlying concept."
            ],
            "weaknesses": [
                "Could elaborate further on production scale trade-offs.",
                "Lacks quantitative benchmarks or specific edge-case handling."
            ],
            "actionable_feedback": [
                "Structure answers with the STAR method or clear pros/cons.",
                "Quantify latency, throughput, or memory implications."
            ],
            "improved_sample_answer": (
                "A comprehensive answer explains the underlying data structure, "
                "details trade-offs such as memory overhead and write amplification, "
                "and illustrates with a concrete production example."
            ),
            "follow_up_question": "How would you monitor and benchmark this implementation in a production environment?"
        }
        return json.dumps(mock_payload, indent=2)
