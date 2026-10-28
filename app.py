#!/usr/bin/env python3
"""
InterviewMate — AI Mock Interviewer Application
===============================================
A web interface built with Gradio for conducting realistic technical,
AI, and behavioural mock interviews powered by an open-weight model
(Qwen2.5-1.5B-Instruct) and fine-tuned LoRA adapter.

Runs directly inside Google Colab (generating a public share link)
or locally in testing mode.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Tuple, Optional

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import gradio as gr
from src.interview_engine import InterviewSession, QUESTION_BANK
from src.model_loader import (
    load_model_and_tokenizer,
    check_gpu_status,
    DEFAULT_MODEL_NAME
)

# Parse command line flags
parser = argparse.ArgumentParser(description="InterviewMate Gradio Application")
parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
                    help="Hugging Face model ID")
parser.add_argument("--adapter", type=str, default=os.getenv("ADAPTER_PATH", None),
                    help="Path to trained LoRA adapter directory")
parser.add_argument("--share", action="store_true", default=os.getenv("SHARE_GRADIO", "false").lower() == "true",
                    help="Create a public temporary Gradio link (ideal for Colab demos)")
parser.add_argument("--port", type=int, default=7860, help="Port to run server on")
args, _ = parser.parse_known_args()

# Global model state (loaded once on startup)
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_ADAPTER_ACTIVE = False
HARDWARE_INFO = check_gpu_status()


def initialize_model():
    """Initializes model and tokenizer once when server boots."""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_ADAPTER_ACTIVE
    if GLOBAL_MODEL is None:
        print("[App] Initializing model engine...")
        GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_ADAPTER_ACTIVE = load_model_and_tokenizer(
            model_name=args.model,
            adapter_path=args.adapter,
            load_in_4bit=True
        )


# ==============================================================================
# GRADIO CONTROLLERS & EVENT HANDLERS
# ==============================================================================

def on_start_interview(role, level, category, num_questions):
    """Initializes a new interview session and displays the first question."""
    session = InterviewSession(
        role=role,
        experience_level=level,
        category=category,
        total_questions=int(num_questions)
    )
    first_q = session.get_current_question()
    topic = session.get_current_topic()

    status_header = (
        f"### 🎯 Question 1 of {session.total_questions}\n"
        f"**Role:** {role} | **Level:** {level} | **Topic:** `{topic}`"
    )

    # Return updated components:
    # (session_state, setup_card_visibility, interview_card_visibility, report_card_visibility,
    #  status_header, question_text, candidate_answer_input, feedback_box, next_btn, finish_btn)
    return (
        session,
        gr.update(visible=False),   # Hide setup
        gr.update(visible=True),    # Show interview
        gr.update(visible=False),   # Hide report
        status_header,
        first_q,
        "",                         # Clear previous answer
        gr.update(visible=False),   # Hide feedback until answer submitted
        gr.update(visible=False),   # Next button hidden until feedback
        gr.update(visible=True)     # Early finish button visible
    )


def on_submit_answer(candidate_answer, session: InterviewSession):
    """Evaluates candidate response, returns immediate rubric feedback."""
    if not session or session.is_completed:
        return (
            session,
            gr.update(visible=False),
            gr.update(visible=False),
            "No active session."
        )

    if not candidate_answer or not candidate_answer.strip():
        # Handle empty answer gracefully
        return (
            session,
            gr.update(
                visible=True,
                value="⚠️ **Please provide an answer before submitting.** Type your response in the box below."
            ),
            gr.update(visible=False),
            ""
        )

    # Evaluate using the model engine
    feedback = session.submit_answer(
        candidate_answer=candidate_answer.strip(),
        model=GLOBAL_MODEL,
        tokenizer=GLOBAL_TOKENIZER
    )

    # Build Markdown feedback card
    score = feedback["overall_score"]
    rubric = feedback["scoring_rubric"]
    strengths_html = "\n".join([f"- ✅ {s}" for s in feedback["strengths"]])
    weaknesses_html = "\n".join([f"- ⚠️ {w}" for w in feedback["weaknesses"]])
    advice_html = "\n".join([f"- 💡 {a}" for a in feedback["actionable_feedback"]])

    badge_color = "#10b981" if score >= 8.0 else ("#f59e0b" if score >= 5.0 else "#ef4444")
    badge_label = "Strong" if score >= 8.0 else ("Average" if score >= 5.0 else "Needs Work")

    feedback_md = f"""
### 📋 Immediate Evaluation & Rubric Feedback

<div style="background-color: {badge_color}; color: white; display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;">
  Score: {score} / 10 ({badge_label})
</div>

#### 📊 Rubric Breakdown
| Metric | Score | Definition |
| :--- | :--- | :--- |
| **Relevance** | **{rubric['relevance']} / 10** | Directness & accuracy in answering the question |
| **Technical Depth** | **{rubric['accuracy_depth']} / 10** | Accuracy, complexity, edge cases, and mechanics |
| **Clarity & Structure** | **{rubric['clarity_structure']} / 10** | Logical flow and structured communication |
| **Completeness** | **{rubric['completeness']} / 10** | Coverage of trade-offs and operational realities |

#### 🌟 Key Strengths
{strengths_html}

#### ⚠️ Areas for Improvement
{weaknesses_html}

#### 💡 Actionable Coaching Advice
{advice_html}

#### 📖 Improved Sample Answer
> *"{feedback['improved_sample_answer']}"*

---
#### 🔄 Contextual Follow-up Question
> **Interviewer asks:** *"{feedback['follow_up_question']}"*
"""

    return (
        session,
        gr.update(visible=True, value=feedback_md),
        gr.update(visible=True),  # Reveal Next Question button
        feedback["follow_up_question"]
    )


def on_next_question(session: InterviewSession):
    """Advances to the next question in the queue or triggers final report if complete."""
    has_next = session.advance_next_question()

    if not has_next:
        # Reached the end! Generate report automatically
        report = session.finish_interview()
        return (
            session,
            gr.update(visible=False),  # Hide interview
            gr.update(visible=True),   # Show report
            report["markdown_report"],
            report["markdown_report"], # For download
            "",                        # Clear status
            "",                        # Clear question
            "",                        # Clear answer
            gr.update(visible=False)   # Hide feedback
        )

    next_q = session.get_current_question()
    topic = session.get_current_topic()
    q_num = session.current_index + 1

    status_header = (
        f"### 🎯 Question {q_num} of {session.total_questions}\n"
        f"**Role:** {session.role} | **Level:** {session.experience_level} | **Topic:** `{topic}`"
    )

    return (
        session,
        gr.update(visible=True),   # Keep interview visible
        gr.update(visible=False),  # Keep report hidden
        "",                        # No report yet
        "",                        # No download yet
        status_header,
        next_q,
        "",                        # Clear answer input
        gr.update(visible=False)   # Hide feedback until answer submitted
    )


def on_finish_interview_early(session: InterviewSession):
    """Allows candidate to conclude the interview at any time and get a performance audit."""
    if not session:
        return gr.update(visible=True), gr.update(visible=False), "No session found.", ""

    report = session.finish_interview()
    return (
        gr.update(visible=False),  # Hide interview
        gr.update(visible=True),   # Show report
        report["markdown_report"],
        report["markdown_report"]  # Downloadable content
    )


def on_reset_to_setup():
    """Returns candidate back to setup screen for a fresh session."""
    return (
        gr.update(visible=True),   # Show setup
        gr.update(visible=False),  # Hide interview
        gr.update(visible=False),  # Hide report
        None                       # Clear session state
    )


# ==============================================================================
# GRADIO UI LAYOUT & STYLES
# ==============================================================================

CUSTOM_CSS = """
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 24px 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.main-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
}
.main-header p {
    margin-top: 8px;
    font-size: 1.05rem;
    opacity: 0.9;
}
.status-pill {
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.9rem;
    margin-bottom: 15px;
}
.question-card {
    background-color: #f8fafc;
    border-left: 5px solid #2563eb;
    padding: 18px;
    border-radius: 8px;
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 16px;
}
"""

def build_gradio_app():
    # Ensure model is ready
    initialize_model()

    model_engine_status = (
        "🚀 LoRA Fine-Tuned Adapter Active" if GLOBAL_ADAPTER_ACTIVE
        else ("⚡ Pretrained Base Model Active" if HARDWARE_INFO["hf_available"]
              else "🛠️ Mock Development Engine Active")
    )
    hardware_desc = (
        f"{HARDWARE_INFO['device_name']} ({HARDWARE_INFO['free_memory_gb']} GB Free VRAM)"
        if HARDWARE_INFO["cuda_available"] else "CPU / Local Mode"
    )

    with gr.Blocks(title="InterviewMate — AI Mock Interviewer") as demo:
        # Hidden session state
        session_state = gr.State(None)

        # Header
        gr.HTML(f"""
        <div class="main-header">
            <h1>🎓 InterviewMate</h1>
            <p>Interview-Specialized AI Mock Interviewer & Comprehensive Coaching System</p>
            <div style="font-size: 0.85em; margin-top: 6px; opacity: 0.85;">
                Engine: <b>{model_engine_status}</b> | Hardware: <b>{hardware_desc}</b>
            </div>
        </div>
        """)

        # ----------------------------------------------------------------------
        # VIEW 1: SETUP CARD
        # ----------------------------------------------------------------------
        with gr.Column(visible=True) as setup_card:
            gr.Markdown("### ⚙️ Step 1: Configure Your Mock Interview")
            with gr.Row():
                role_dropdown = gr.Dropdown(
                    label="Target Job Role",
                    choices=[
                        "Software Engineer",
                        "Backend Engineer",
                        "Frontend Developer",
                        "Fullstack Developer",
                        "Machine Learning Engineer",
                        "Data Scientist",
                        "AI / NLP Engineer",
                        "DevOps / Cloud Engineer",
                        "System Architect"
                    ],
                    value="Software Engineer"
                )
                level_dropdown = gr.Dropdown(
                    label="Experience Level",
                    choices=[
                        "Entry-Level / Junior (0-2 years)",
                        "Mid-Level (2-5 years)",
                        "Senior / Lead (5+ years)"
                    ],
                    value="Mid-Level (2-5 years)"
                )

            with gr.Row():
                category_radio = gr.Radio(
                    label="Interview Track",
                    choices=[
                        ("💻 Software Development", "software_development"),
                        ("🤖 AI & Data Science", "ai_data_science"),
                        ("🤝 HR & Behavioural (STAR)", "hr_behavioural")
                    ],
                    value="software_development"
                )
                num_questions_slider = gr.Slider(
                    label="Number of Questions",
                    minimum=2,
                    maximum=6,
                    value=3,
                    step=1
                )

            start_btn = gr.Button("🚀 Start Interview Session", variant="primary", size="lg")

            with gr.Accordion("ℹ️ Candidate Guidance & Scoring Rubric", open=False):
                gr.Markdown("""
                - **How it works:** You will be presented with one interview question at a time.
                - **Feedback:** As soon as you submit your response, the AI evaluates your answer against an objective 4-part rubric: *Relevance*, *Technical Depth*, *Clarity & Structure*, and *Completeness*.
                - **Follow-ups:** The AI asks contextual follow-up questions tailored to your answer.
                - **Final Report:** At the end of the session, a complete performance audit and a personalized 7-day preparation plan are generated.
                """)

        # ----------------------------------------------------------------------
        # VIEW 2: ACTIVE INTERVIEW CARD
        # ----------------------------------------------------------------------
        with gr.Column(visible=False) as interview_card:
            status_header_md = gr.Markdown("### 🎯 Question 1 of 3")

            question_display = gr.Markdown(
                label="Interviewer Question",
                value="",
                elem_classes=["question-card"]
            )

            with gr.Row():
                candidate_answer_input = gr.Textbox(
                    label="Your Answer",
                    placeholder="Type your response thoroughly. Explain underlying principles, trade-offs, or use the STAR method for behavioural questions...",
                    lines=6,
                    max_lines=12
                )

            with gr.Row():
                submit_answer_btn = gr.Button("📤 Submit Answer for Feedback", variant="primary")
                finish_early_btn = gr.Button("⏹️ Conclude & View Final Report", variant="secondary")

            # Feedback panel
            feedback_output_md = gr.Markdown(visible=False)

            # Navigation buttons
            with gr.Row():
                next_question_btn = gr.Button("➡️ Proceed to Next Question", variant="primary", visible=False)

        # ----------------------------------------------------------------------
        # VIEW 3: FINAL REPORT CARD
        # ----------------------------------------------------------------------
        with gr.Column(visible=False) as report_card:
            report_display_md = gr.Markdown("### 🎓 Final Performance Report")

            with gr.Row():
                restart_btn = gr.Button("🔄 Start Another Mock Interview", variant="primary")
                download_report_btn = gr.DownloadButton(
                    label="📥 Download Performance Report (Markdown)",
                    visible=True
                )

        # ----------------------------------------------------------------------
        # EVENT WIRING
        # ----------------------------------------------------------------------
        # Start button
        start_btn.click(
            fn=on_start_interview,
            inputs=[role_dropdown, level_dropdown, category_radio, num_questions_slider],
            outputs=[
                session_state,
                setup_card,
                interview_card,
                report_card,
                status_header_md,
                question_display,
                candidate_answer_input,
                feedback_output_md,
                next_question_btn,
                finish_early_btn
            ]
        )

        # Submit answer button
        submit_answer_btn.click(
            fn=on_submit_answer,
            inputs=[candidate_answer_input, session_state],
            outputs=[
                session_state,
                feedback_output_md,
                next_question_btn,
                gr.State() # Discard follow-up return string
            ]
        )

        # Next question button
        next_question_btn.click(
            fn=on_next_question,
            inputs=[session_state],
            outputs=[
                session_state,
                interview_card,
                report_card,
                report_display_md,
                download_report_btn,
                status_header_md,
                question_display,
                candidate_answer_input,
                feedback_output_md
            ]
        )

        # Conclude early button
        finish_early_btn.click(
            fn=on_finish_interview_early,
            inputs=[session_state],
            outputs=[
                interview_card,
                report_card,
                report_display_md,
                download_report_btn
            ]
        )

        # Restart button
        restart_btn.click(
            fn=on_reset_to_setup,
            inputs=[],
            outputs=[
                setup_card,
                interview_card,
                report_card,
                session_state
            ]
        )

    return demo


if __name__ == "__main__":
    app = build_gradio_app()
    print(f"\n[InterviewMate] Launching Gradio interface on port {args.port} (share={args.share})...")
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )
