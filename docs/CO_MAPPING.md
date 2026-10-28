# 🎯 Course Outcome (CO) Mapping & Deep Learning Curriculum Alignment

**Course Code:** 24ADI006  
**Course Name:** Deep Learning  
**Project Title:** InterviewMate – AI Mock Interviewer  
**Project Category:** Project Based  
**Student Name:** jeevaanantham  
**Roll Number:** 24BAD047  
**Academic Repository:** [https://github.com/jeevaa006/InterviewMate](https://github.com/jeevaa006/InterviewMate)  
**Primary Course Outcome:** **CO5**  
**Supporting Course Outcomes:** **CO1** and **CO3**  

---

## 📋 Executive Summary & Abstract

**InterviewMate** is an intelligent, domain-specialized mock interview system powered by an open-weight Transformer-based Large Language Model ([`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)) fine-tuned using parameter-efficient fine-tuning (**4-bit QLoRA**). The application addresses the barrier students and entry-level job seekers face in accessing personalized, technical interview practice and structured feedback without depending on costly closed-source cloud APIs.

The system evaluates candidate answers across four objective rubric pillars (Relevance, Technical Accuracy & Depth, Clarity & Structure, and Completeness), extracts structured analytical feedback and actionable advice, generates context-sensitive follow-up questions, and renders an interactive web demonstration via Gradio.

This document presents a rigorous, honest mapping of InterviewMate's architecture, methodology, codebase, and training pipeline to the formal Course Outcomes (CO1–CO5) of **24ADI006 – Deep Learning**.

> [!NOTE]
> **Academic Alignment Statement:**  
> The alignments presented herein represent self-contained, project-level technical mappings demonstrating how the software and machine learning engineering artifacts satisfy course concepts. They do not constitute a pre-awarded or official faculty assessment.

---

## 📊 Complete Course Outcome Mapping Matrix

The curriculum for **24ADI006 – Deep Learning** defines five core Course Outcomes:

| Course Outcome | Syllabus Definition | Project Alignment Status | Primary Justification Summary | Key Repository Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **CO1** | *Apply neural network architectures, activation functions, optimization methods, and regularization techniques to develop and evaluate deep learning models for predictive tasks.* | **Supporting Mapping** | Supervised fine-tuning via Hugging Face `TRL SFTTrainer` applying **Paged AdamW 8-bit** optimizer, **Cosine Learning Rate scheduling**, dropout-based LoRA regularization, cross-entropy causal language modeling loss, and Mean Absolute Error (MAE) evaluation against expert rubric labels. | [`colab/InterviewMate_Training_and_Demo.ipynb`](../colab/InterviewMate_Training_and_Demo.ipynb#L18), [`src/evaluation.py`](../src/evaluation.py#L9-L12), [`scripts/generate_dataset.py`](../scripts/generate_dataset.py#L34-L36) |
| **CO2** | *Implement deep learning techniques for feature extraction, image analysis, and computer vision applications.* | **Not Directly Mapped** | InterviewMate is purely focused on Natural Language Processing (NLP) and speech-to-text transcript processing. It does not employ Convolutional Neural Networks (CNNs), Vision Transformers (ViT), image segmentation, or spatial feature extraction pipelines. | *(Explicitly scoped out of project domain)* |
| **CO3** | *Develop sequential learning models for processing and analyzing sequential and natural language data.* | **Supporting Mapping** | Processes sequential token streams and contextual natural language interview dialogs. Implements tokenization, positional encodings (RoPE), and causal autoregressive sequence modeling. Acknowledges the modern shift from recurrent units (RNN/LSTM/GRU) to self-attention-based sequence processors. | [`src/feedback.py`](../src/feedback.py#L29-L81), [`src/model_loader.py`](../src/model_loader.py#L95-L112), [`data/interview_train.jsonl`](../data/interview_train.jsonl) |
| **CO4** | *Analyze generative and unsupervised deep learning techniques for representation learning, feature extraction, and synthetic data generation.* | **Contextual / Related Connection** | Leverages autoregressive text generation to produce candidate coaching feedback and follow-up inquiries. Also utilizes a rule-based synthetic dataset generation pipeline. However, generating text using an instruction-tuned decoder **does not** constitute implementation of Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion models, or self-supervised contrastive representation learning. | [`scripts/generate_dataset.py`](../scripts/generate_dataset.py#L1-L26), [`src/feedback.py`](../src/feedback.py#L124-L162) |
| **CO5** | *Analyze attention-based architectures and advanced deep learning techniques to develop intelligent applications for language and multimodal AI tasks.* | **Primary Mapping** | Implements and adapts a state-of-the-art Transformer language model (`Qwen2.5-1.5B-Instruct`) utilizing **Scaled Dot-Product Multi-Head Attention**, **Grouped-Query Attention (GQA)**, and **Rotary Position Embedding (RoPE)**. Employs advanced parameter-efficient deep learning (**QLoRA** with 4-bit NormalFloat quantization and Low-Rank Adaptation) and serves an interactive real-time language application via Gradio. | [`src/model_loader.py`](../src/model_loader.py#L62-L131), [`src/interview_engine.py`](../src/interview_engine.py#L1-L30), [`colab/InterviewMate_Training_and_Demo.ipynb`](../colab/InterviewMate_Training_and_Demo.ipynb#L16), [`app.py`](../app.py#L50-L60) |

---

## 🔍 Detailed Course Outcome Justifications

### 1. Primary Mapping: CO5 — Attention Architectures & Advanced Language Applications

> **Syllabus CO5:** *Analyze attention-based architectures and advanced deep learning techniques to develop intelligent applications for language and multimodal AI tasks.*

#### Why CO5 is the Primary Outcome:
1. **Underlying Transformer Architecture:**
   InterviewMate's core engine relies directly on the **Decoder-Only Transformer** architecture instantiated by `Qwen/Qwen2.5-1.5B-Instruct`. This architecture uses:
   - **Scaled Dot-Product Self-Attention:**
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
     allowing every token in the candidate's answer to dynamically attend to relevant tokens across the interview prompt and rubric guidelines.
   - **Grouped-Query Attention (GQA):** Employs fewer key-value heads than query heads, reducing KV cache memory bandwidth during interactive token generation.
   - **Rotary Position Embeddings (RoPE):** Encodes relative token distances directly into query-key inner products, preserving context across up to 32,768 tokens.
   - **SwiGLU Non-Linear Feed-Forward Projections:** Enhances representational expressiveness in the MLP sublayers.

2. **Advanced Deep Learning Techniques (QLoRA):**
   Rather than treating the foundation model as an opaque black box accessed via an external web API, InterviewMate implements advanced **Parameter-Efficient Fine-Tuning (PEFT)**:
   - **4-Bit NormalFloat (NF4) Quantization:** Quantizes frozen weights to an information-theoretically optimal distribution for Gaussian-distributed neural network tensors ([`src/model_loader.py`](../src/model_loader.py#L62-L78)).
   - **Double Quantization (DQ):** Quantizes the first-order quantization scale constants, saving ~0.37 bits per parameter.
   - **Low-Rank Adaptation ($\Delta W = \frac{\alpha}{r} B \cdot A$):** Injects trainable low-rank decomposition rank matrices ($r=16, \alpha=32$) into all linear attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

3. **Intelligent End-to-End Application:**
   The trained network is not an isolated academic exercise; it powers a complete conversational system (`app.py`) that executes real-time inference, parses complex structured outputs, dynamically handles multi-turn follow-up inquiries, and outputs hiring readiness reports.

---

### 2. Supporting Mapping: CO1 — Neural Network Optimization, Regularization, and Evaluation

> **Syllabus CO1:** *Apply neural network architectures, activation functions, optimization methods, and regularization techniques to develop and evaluate deep learning models for predictive tasks.*

#### Why CO1 is Directly Supported:
1. **Optimization Methods:**
   - **Paged AdamW 8-bit:** The fine-tuning script (`colab/InterviewMate_Training_and_Demo.ipynb`) configures `paged_adamw_8bit`. This optimizer dynamically pages optimizer state tensors between GPU VRAM and CPU system RAM when memory allocation spikes occur during gradient backpropagation.
   - **Learning Rate Scheduling:** Implements a **Cosine Annealing Learning Rate Scheduler** with a linear warmup phase (`warmup_steps=10`, initial peak $\eta = 2 \times 10^{-4}$), mitigating catastrophic forgetting of base linguistic capabilities.
   - **Gradient Accumulation:** Employs gradient accumulation (`gradient_accumulation_steps=4`, batch size 2) to simulate an effective batch size of 8 on a memory-constrained Tesla T4 GPU.

2. **Regularization Techniques:**
   - **LoRA Dropout ($p=0.05$):** Applied to the input activations of low-rank matrices $A$ to prevent overfitting on the fine-tuning dataset.
   - **Frozen Base Weights:** Over 99% of network parameters remain frozen, acting as an implicit prior that prevents weight drift and degradation of general reasoning capabilities.

3. **Evaluation for Predictive & Scoring Tasks:**
   - The system evaluates rubric prediction using **Mean Absolute Error (MAE)** between predicted integer/float scores and verified rubric ground truths across held-out evaluation scenarios ([`src/evaluation.py`](../src/evaluation.py#L80-L101)).
   - Evaluates cross-entropy loss tracking across training epochs via Hugging Face `SFTTrainer`.

---

### 3. Supporting Mapping: CO3 — Sequential Learning & Natural Language Processing

> **Syllabus CO3:** *Develop sequential learning models for processing and analyzing sequential and natural language data.*

#### Why CO3 is Supported:
1. **Sequential Nature of Language Data:**
   - Natural language dialog in mock interviews is strictly sequential and causal. The model autoregressively conditions each token $y_t$ on the entire preceding context history $(x_1, \dots, x_N, y_1, \dots, y_{t-1})$.
   - Implements causal attention masking, ensuring position $i$ cannot access future tokens $j > i$.

2. **Modern Evolution from Recurrent Units:**
   - While traditional Deep Learning curricula introduce sequential modeling via Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRUs), modern sequential learning is dominated by causal autoregressive Transformers.
   - InterviewMate explicitly demonstrates this architectural progression: solving long-range sequential dependencies and vanishing gradients through self-attention rather than sequential hidden state recurrence.

3. **Multi-turn Contextual Progression:**
   - The interview engine tracks sequential interview turns, maintaining conversation history and generating follow-up inquiries conditioned on the candidate's specific prior responses ([`src/interview_engine.py`](../src/interview_engine.py#L182-L215)).

---

### 4. Contextual Outcome: CO4 — Generative AI vs Unsupervised Deep Learning

> **Syllabus CO4:** *Analyze generative and unsupervised deep learning techniques for representation learning, feature extraction, and synthetic data generation.*

#### Honest Clarification Regarding CO4:
- **Connection to Generative AI:** InterviewMate generates novel natural language text (rubric breakdowns, constructive critiques, model answers, follow-up questions) and utilizes a synthetic data generation framework ([`scripts/generate_dataset.py`](../scripts/generate_dataset.py)) governed by expert rubrics.
- **Why it is NOT Claimed as a Core Course Outcome:**  
  Generating text using an autoregressive, instruction-tuned Transformer does not satisfy the core requirements of classical CO4 topics:
  - No **Variational Autoencoder (VAE)** or encoder-decoder bottleneck architecture is trained for latent representation learning.
  - No **Generative Adversarial Network (GAN)** with discriminator minimax optimization is implemented.
  - No **Diffusion Model** with forward noise addition and reverse denoising score matching is utilized.
  - No unsupervised representation learning (e.g., contrastive learning like SimCLR or masked autoencoding) was implemented from scratch.
- Therefore, CO4 is documented honestly as a related conceptual domain, rather than an assessed Course Outcome.

---

### 5. Scoped-Out Outcome: CO2 — Computer Vision and Image Analysis

> **Syllabus CO2:** *Implement deep learning techniques for feature extraction, image analysis, and computer vision applications.*

#### Clear Academic Demarcation:
- InterviewMate is strictly a text- and speech-transcript-based conversational natural language processing system.
- It does not process video frames, facial expressions, eye-tracking vectors, or spatial image tensors.
- Conforming with academic integrity, **CO2 is explicitly documented as unmapped**.

---

## 🗂️ Codebase Evidence & Verification Index

| Course Outcome Element | Project Artifact | Specific File / Module | Function / Class / Configuration |
| :--- | :--- | :--- | :--- |
| **Attention Mechanism & Transformer Core (CO5)** | Base Language Model | [`src/model_loader.py`](../src/model_loader.py) | `load_model_and_tokenizer()`, `AutoModelForCausalLM` with `trust_remote_code=True` |
| **Quantization & PEFT Architecture (CO5)** | QLoRA Engine | [`src/model_loader.py`](../src/model_loader.py#L62-L78), [`colab/InterviewMate_Training_and_Demo.ipynb`](../colab/InterviewMate_Training_and_Demo.ipynb#L16) | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`, `LoraConfig(r=16, lora_alpha=32)` |
| **Interactive Language Application (CO5)** | Gradio Web App | [`app.py`](../app.py#L66-L165), [`src/interview_engine.py`](../src/interview_engine.py#L125-L225) | `InterviewSession`, `on_submit_answer()`, `on_next_question()` |
| **Optimization & Hyperparameters (CO1)** | Colab Training Pipeline | [`colab/InterviewMate_Training_and_Demo.ipynb`](../colab/InterviewMate_Training_and_Demo.ipynb#L18) | `SFTConfig(optim="paged_adamw_8bit", lr_scheduler_type="cosine", learning_rate=2e-4, num_train_epochs=3)` |
| **Predictive Scoring & Loss Evaluation (CO1)** | Evaluation Benchmark | [`src/evaluation.py`](../src/evaluation.py#L51-L101), [`src/feedback.py`](../src/feedback.py#L124-L162) | `evaluate_single_sample()`, `run_evaluation_benchmark()`, Score MAE Calculation |
| **Unit & Integration Verification (CO1)** | Automated Tests | [`tests/test_interviewmate.py`](../tests/test_interviewmate.py#L34-L150) | `TestDatasetIntegrity`, `TestModelLoaderAndHardware`, `TestFeedbackAndJSONParser` (12 automated unit tests) |
| **Sequential Prompting & Token Formatting (CO3)** | ChatML Formatter | [`colab/InterviewMate_Training_and_Demo.ipynb`](../colab/InterviewMate_Training_and_Demo.ipynb#L14), [`src/feedback.py`](../src/feedback.py#L29-L81) | `apply_chat_template()`, `build_evaluation_prompt()` |
| **Multi-Turn Context History (CO3)** | Session Progression | [`src/interview_engine.py`](../src/interview_engine.py#L170-L215) | Multi-question state tracking, contextual follow-up generation |

---

## 🔬 Experimental Results & Verification Status

To ensure complete academic transparency, experimental results within InterviewMate are categorized into verified repository evidence and live runtime executions:

1. **Verified Static Repository Evidence:**
   - **Dataset Volume:** Exactly 156 records generated and partitioned ([`data/interview_train.jsonl`](../data/interview_train.jsonl) = 123 records; [`data/interview_eval.jsonl`](../data/interview_eval.jsonl) = 33 records).
   - **Zero Leakage:** Confirmed by automated test `test_zero_data_leakage` in [`tests/test_interviewmate.py`](../tests/test_interviewmate.py#L71-L80) using scenario prefix grouping.
   - **Defensive Parser Stability:** 100% test pass rate across 12 automated unit tests verifying JSON extraction, trailing comma repairs, and fallback heuristic behavior.
   - **VRAM Footprint:** Base model memory is confirmed at ~1.2 GB in 4-bit NF4 representation, verified on Tesla T4 architectures.

2. **Live Runtime Executions (To Be Verified in Colab):**
   - Fine-tuning loss curve, final checkpoint weights (`adapter_model.safetensors`), and comparative evaluation latency/MAE are generated dynamically during the Google Colab execution run.
   - Checkpoints are saved locally to `saved_adapter/` and optionally backed up to Google Drive, adhering to git hygiene standards that prohibit committing multi-gigabyte binary weights to source control.

---

## 🏁 Conclusion

InterviewMate demonstrates a cohesive, practical application of modern deep learning principles to solve a high-impact educational problem. By combining **Transformer self-attention architectures (CO5)**, **parameter-efficient optimization and regularization (CO1)**, and **sequential natural language processing (CO3)**, the project illustrates how open-weight foundation models can be economically adapted on accessible cloud hardware to create dependable, real-time AI applications.
