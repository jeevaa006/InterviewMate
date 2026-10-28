# 📹 Individual Video Presentation & Walkthrough Guide

**Course Code:** 24ADI006  
**Course Name:** Deep Learning  
**Project Title:** InterviewMate – AI Mock Interviewer  
**Student Name:** jeevaanantham  
**Roll Number:** 24BAD047  
**Academic Repository:** [https://github.com/jeevaa006/InterviewMate](https://github.com/jeevaa006/InterviewMate)  
**Target Video Duration:** 5 to 7 Minutes  

---

## 🔗 Shareable Video Submission Link

```text
VIDEO SUBMISSION URL:
PENDING — Add final shareable URL (e.g., Unlisted YouTube Link or Public Google Drive URL)
```

> [!IMPORTANT]
> **Instructions for the Student (`jeevaaanantham`):**
> 1. Record your screen and camera walkthrough following the script and timestamps provided below.
> 2. Upload the MP4 video to **YouTube** (set visibility to *Unlisted*) or **Google Drive** (ensure link sharing is set to *"Anyone with the link can view"*).
> 3. Replace the placeholder text above with your live URL:
>    ```text
>    VIDEO SUBMISSION URL: https://youtu.be/your-video-id
>    ```
> 4. Update the corresponding link in [`README.md`](../README.md) and commit the change to your GitHub repository.

---

## ⏱️ Video Presentation Script & Timeline (5–7 Minutes)

This presentation script is structured into 8 distinct segments designed to provide faculty evaluators with a thorough, well-paced explanation of the theoretical foundations, implementation architecture, live demonstration, and Course Outcome alignments.

```
Total Duration: ~6:00 – 7:00 Minutes
┌────────────────┬─────────────────────────────────────────────────────────────┬──────────┐
│ Timestamp      │ Presentation Section                                        │ Duration │
├────────────────┼─────────────────────────────────────────────────────────────┼──────────┤
│ 00:00 – 00:45  │ 1. Student Introduction & Problem Statement                 │ 0:45 min │
│ 00:45 – 01:30  │ 2. Project Objectives & Core Capabilities                   │ 0:45 min │
│ 01:30 – 02:30  │ 3. System Architecture & End-to-End Workflow                │ 1:00 min │
│ 02:30 – 03:30  │ 4. Base Model & Transformer Self-Attention Concepts         │ 1:00 min │
│ 03:30 – 04:45  │ 5. QLoRA Fine-Tuning, Dataset & Training Pipeline           │ 1:15 min │
│ 04:45 – 06:00  │ 6. Live Application Walkthrough & Rubric Demonstration      │ 1:15 min │
│ 06:00 – 06:45  │ 7. Deep Learning Course Outcome (CO) Mapping (Focus on CO5) │ 0:45 min │
│ 06:45 – 07:15  │ 8. Engineering Safeguards, Limitations & Conclusion         │ 0:30 min │
└────────────────┴─────────────────────────────────────────────────────────────┴──────────┘
```

---

### Segment 1: Student Introduction & Problem Statement (00:00 – 00:45)

**Visual:** Webcam on student with title slide displaying:
- *InterviewMate – AI Mock Interviewer*
- *Course: 24ADI006 – Deep Learning*
- *Student: jeevaanantham (Roll No: 24BAD047)*

**Spoken Script:**
> *"Respected faculty and evaluators, good day. My name is Jeevaanantham, Roll Number 24BAD047, presenting my Deep Learning capstone project for course 24ADI006: **InterviewMate – AI Mock Interviewer**.*
>
> *Students and junior software engineers face a major hurdle when preparing for competitive technical and behavioural interviews: high-quality mock interview coaching is either expensive, inaccessible, or relies on opaque closed-source proprietary APIs. On the other hand, generic off-the-shelf small language models often produce inconsistent, uncalibrated feedback with erratic output formatting.*
>
> *InterviewMate solves this by adapting an open-weight 1.5-billion parameter Transformer language model using 4-bit QLoRA fine-tuning on free cloud GPU resources to serve as an automated, rubric-driven mock interviewer."*

---

### Segment 2: Project Objectives & Core Capabilities (00:45 – 01:30)

**Visual:** Slide highlighting the four primary objectives and the Gradio web interface screenshot.

**Spoken Script:**
> *"Our project was engineered around four specific objectives:*
> 1. *First, to adapt an open-weight foundation model—specifically `Qwen2.5-1.5B-Instruct`—without paying for commercial cloud APIs.*
> 2. *Second, to build a curated, leakage-free synthetic dataset covering Software Engineering, AI & Data Science, and Behavioural STAR-method interview scenarios.*
> 3. *Third, to fine-tune the model with 4-bit QLoRA on a single consumer-tier cloud GPU—the Google Colab Tesla T4—staying well within its 15 gigabytes of VRAM.*
> 4. *And fourth, to integrate the model with a defensive, crash-resilient parsing engine and an interactive Gradio web application for real-time interview practice."*

---

### Segment 3: System Architecture & End-to-End Workflow (01:30 – 02:30)

**Visual:** Screen sharing the architecture flowchart from the repository (`README.md` / `docs/PROJECT_REPORT.md`).

**Spoken Script:**
> *"Let us examine the system architecture. The pipeline consists of four integrated layers:*
> - *First, the **Data Pipeline**: Our generator script builds 156 validated question-answer pairs. We apply scenario-level group partitioning, ensuring 100% held-out separation between our 123 training records and 33 evaluation records with zero leakage.*
> - *Second, the **Deep Learning & Fine-Tuning Pipeline**: Hosted on Google Colab, the base model is quantized to 4-bit NormalFloat using `bitsandbytes`, and low-rank LoRA adapters are attached across all linear projections. We train using Hugging Face's `TRL SFTTrainer`.*
> - *Third, the **Evaluation & Defensive Parser**: The model's generated response is intercepted by `src/feedback.py`, which uses a multi-tier regex and JSON repair engine to guarantee valid rubric extraction even if the model outputs formatting quirks.*
> - *And fourth, the **Serving Interface**: An interactive web UI built with Gradio Blocks running with public shareable links, allowing any candidate to take a realistic mock interview from their browser."*

---

### Segment 4: Base Model & Transformer Self-Attention Concepts (02:30 – 03:30)

**Visual:** Slide or notebook showing `Qwen/Qwen2.5-1.5B-Instruct` architecture, self-attention formulas, and tokenization.

**Spoken Script:**
> *"At the mathematical core of InterviewMate is the Transformer architecture, connecting directly to **Course Outcome 5**.*
>
> *We selected `Qwen2.5-1.5B-Instruct`, an Apache 2.0 open-weight decoder model. The model processes text sequences using multi-head Scaled Dot-Product Attention:*
>
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
>
> *Here, queries, keys, and values are linearly projected from the token embeddings. The division by $\sqrt{d_k}$ prevents the dot products from growing excessively large in high dimensions, which would otherwise push the softmax function into regions with vanishing gradients.*
>
> *Furthermore, Qwen2.5 incorporates Rotary Position Embeddings (RoPE), enabling long-range context modeling across conversation turns, and Grouped-Query Attention (GQA), which significantly reduces the key-value cache memory footprint during autoregressive decoding."*

---

### Segment 5: QLoRA Fine-Tuning, Dataset & Training Pipeline (03:30 – 04:45)

**Visual:** Switching to `colab/InterviewMate_Training_and_Demo.ipynb`, scrolling through the dataset loading, BitsAndBytes configuration, and `SFTTrainer` cells.

**Spoken Script:**
> *"Now let us look at our training notebook in Google Colab. Full fine-tuning of a 1.5 billion parameter model in 16-bit float requires over 16 gigabytes of VRAM just for weights and optimizer states, which would crash a standard Colab Tesla T4 GPU with an Out-of-Memory error.*
>
> *To solve this, we applied **QLoRA**:*
> 1. *We freeze the base model and quantize its parameters into **4-bit NormalFloat (NF4)**, reducing base weight memory to approximately 1.2 gigabytes.*
> 2. *We enable **Double Quantization**, quantizing the quantization scaling constants themselves.*
> 3. *We inject low-rank decomposition matrices $\Delta W = \frac{\alpha}{r} (B \cdot A)$ with rank $r=16$ and scaling $\alpha=32$ across all attention and MLP layers.*
> 4. *We use the **Paged AdamW 8-bit** optimizer, which pages memory to system RAM during gradient spikes, combined with a **Cosine Annealing** learning rate scheduler starting at $2 \times 10^{-4}$ over 3 training epochs.*
>
> *This allows complete fine-tuning in under 18 minutes on free Colab tier hardware, consuming less than 6 gigabytes of peak VRAM."*

---

### Segment 6: Live Application Walkthrough & Rubric Demonstration (04:45 – 06:00)

**Visual:** Live screen recording of the Gradio interface in action (`http://localhost:7860` or Colab public link).

**Spoken Script:**
> *"Here is the live InterviewMate web interface.*
>
> *1. We begin on the Setup screen. We select the target role as 'Backend Engineer', seniority as 'Mid-Level', interview track as 'Software Development', and choose a 2-question session.*
>
> *2. Clicking 'Start Interview Session' transitions us to the interview workspace. The first question is presented: 'How do database indexes improve query performance, and what are the trade-offs of using them?'*
>
> *3. Let us submit a deliberately average response: 'Indexes use B-trees to make searches faster, but they take up extra disk space and slow down writes.'*
>
> *4. We click 'Submit Answer for Evaluation'. In real time, the model scores the response across our 4-pillar rubric:*
> - *Relevance: 8/10*
> - *Technical Depth: 6/10*
> - *Clarity: 7/10*
> - *Completeness: 6/10*
>
> *Notice that InterviewMate immediately delivers constructive feedback: it praises the mention of B-trees and write penalties, flags the omission of composite index rules and cardinality, and supplies a polished, exemplary model answer along with a dynamic follow-up question: 'How does column cardinality influence your decision when creating a composite index?'*
>
> *5. Upon completing questions, the system generates a hiring readiness audit, complete with an overall readiness tier and a 7-day targeted study roadmap, all downloadable as a Markdown report."*

---

### Segment 7: Deep Learning Course Outcome (CO) Mapping (06:00 – 06:45)

**Visual:** Displaying the Course Outcome summary table from `docs/CO_MAPPING.md`.

**Spoken Script:**
> *"Regarding academic alignment with course **24ADI006 – Deep Learning**:*
> - *Our **Primary Outcome is CO5**: We have analyzed and adapted modern attention-based Transformer architectures and advanced deep learning techniques (4-bit QLoRA and PEFT) to construct an intelligent language application.*
> - *Our **Supporting Outcomes are CO1 and CO3**: CO1 is demonstrated through our optimization pipeline (Paged AdamW 8-bit, Cosine learning rate scheduling, LoRA regularization, and rubric scoring MAE evaluation). CO3 is demonstrated through sequential token modeling, causal masking, and conversation-turn contextual handling.*
> - *We explicitly clarify that **CO2 (Computer Vision)** is not mapped as this is an NLP project, and **CO4** is related through generative text modeling rather than unsupervised VAE or GAN implementations."*

---

### Segment 8: Engineering Safeguards, Limitations & Conclusion (06:45 – 07:15)

**Visual:** Displaying test suite execution terminal (`Ran 12 tests in 0.061s ... OK`) and repository summary.

**Spoken Script:**
> *"To ensure engineering robustness, we built a comprehensive test suite of 12 automated unit tests in `tests/test_interviewmate.py`, verifying zero dataset leakage, hardware fallback stability, and defensive multi-tier JSON parsing.*
>
> *Acknowledged limitations include the synthetic nature of the 156-sample dataset and reliance on textual evaluation rather than voice pitch analysis. Future work will integrate Whisper for multimodal audio input and expand human calibration.*
>
> *In conclusion, InterviewMate demonstrates that student-accessible cloud hardware combined with modern parameter-efficient deep learning can deliver practical, reliable, and high-impact AI tools. Thank you for your time and evaluation."*

---

## 💡 Practical Recording Tips for the Student

1. **Audio & Camera:** Ensure your microphone is clear and background noise is minimal. Keep your webcam active in a corner of the screen for faculty verification.
2. **Resolution:** Record at standard 1080p (1920x1080) resolution for crisp presentation of code and notebook cells.
3. **Pacing:** Practice speaking smoothly and do not rush. The script is timed for a calm 130–140 words per minute.
4. **Colab / Local Demo:** You can run the live demo either locally via `python3 app.py --port 7860` (using the built-in mock fallback for instant responses) or in Google Colab with the GPU active.
