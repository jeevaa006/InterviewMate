"""
InterviewMate Interview Engine
==============================
Coordinates the mock interview flow:
- Question bank selection by Track, Role, and Seniority
- Progression (one question at a time)
- Contextual follow-up generation
- State management and scoring history
- Integration with base model and fine-tuned LoRA adapter
"""

import os
import sys
import random
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feedback import (
    SYSTEM_INSTRUCTION,
    build_evaluation_prompt,
    extract_json_from_text,
    sanitize_feedback,
    generate_final_report
)
from src.model_loader import generate_model_response


# Curated question pool mapped by category
QUESTION_BANK: Dict[str, List[Dict[str, Any]]] = {
    "software_development": [
        {
            "id": "SWE_01",
            "question": "How do database indexes improve query performance, and what are the trade-offs of using them?",
            "topic": "Databases & Indexing",
            "suggested_role": "Backend Engineer"
        },
        {
            "id": "SWE_02",
            "question": "What are the fundamental differences between REST and GraphQL APIs, and when would you choose one over the other?",
            "topic": "API Design & Architecture",
            "suggested_role": "Fullstack Developer"
        },
        {
            "id": "SWE_03",
            "question": "What is a race condition, and what strategies do you employ in distributed systems to prevent it?",
            "topic": "Concurrency & Distributed Systems",
            "suggested_role": "Backend Engineer"
        },
        {
            "id": "SWE_04",
            "question": "Explain the CAP theorem and describe how modern distributed databases handle network partitions.",
            "topic": "System Design & Distributed Data",
            "suggested_role": "System Architect"
        },
        {
            "id": "SWE_05",
            "question": "What is the difference between Continuous Delivery and Continuous Deployment, and how do you implement zero-downtime releases?",
            "topic": "CI/CD & DevOps",
            "suggested_role": "DevOps / Cloud Engineer"
        },
        {
            "id": "SWE_06",
            "question": "What caching strategies (Cache-Aside, Write-Through, Write-Behind) do you use with Redis, and how do you handle invalidation?",
            "topic": "Caching & Performance",
            "suggested_role": "Backend Engineer"
        },
        {
            "id": "SWE_07",
            "question": "What is the difference between Docker containerization and Virtual Machines, and why do containers start faster?",
            "topic": "Containerization & Linux Kernels",
            "suggested_role": "DevOps Engineer"
        },
        {
            "id": "SWE_08",
            "question": "How would you design a scalable URL shortening service like Bitly handling 100M new URLs per month?",
            "topic": "High-Level System Design",
            "suggested_role": "Fullstack Developer"
        },
        {
            "id": "SWE_09",
            "question": "What are Core Web Vitals (LCP, INP, CLS), and how do you optimize a modern single-page web application for them?",
            "topic": "Frontend Web Performance",
            "suggested_role": "Frontend Developer"
        },
        {
            "id": "SWE_10",
            "question": "What is HTTP method idempotency, and why are POST, PUT, and DELETE handled differently in distributed architectures?",
            "topic": "Web Protocols & Resilience",
            "suggested_role": "Backend Engineer"
        }
    ],
    "ai_data_science": [
        {
            "id": "AI_01",
            "question": "What is overfitting in machine learning, how do you detect it, and what techniques do you use to mitigate it?",
            "topic": "ML Fundamentals & Regularization",
            "suggested_role": "Machine Learning Engineer"
        },
        {
            "id": "AI_02",
            "question": "How does the self-attention mechanism work in Transformers, and why do we divide by sqrt(d_k)?",
            "topic": "Deep Learning & Transformer Math",
            "suggested_role": "AI / NLP Engineer"
        },
        {
            "id": "AI_03",
            "question": "Why is accuracy a misleading metric for imbalanced classification problems, and what alternative metrics should be used?",
            "topic": "Model Evaluation & Metrics",
            "suggested_role": "Data Scientist"
        },
        {
            "id": "AI_04",
            "question": "What is LoRA (Low-Rank Adaptation), how does QLoRA improve upon it, and why does it enable fine-tuning LLMs on consumer GPUs?",
            "topic": "Parameter-Efficient Fine-Tuning (PEFT)",
            "suggested_role": "LLM / AI Engineer"
        },
        {
            "id": "AI_05",
            "question": "Explain the bias-variance trade-off in machine learning and how it relates to underfitting and overfitting.",
            "topic": "Statistical Learning Theory",
            "suggested_role": "Data Scientist"
        },
        {
            "id": "AI_06",
            "question": "When building an enterprise LLM solution, how do you decide between Retrieval-Augmented Generation (RAG) and Fine-Tuning?",
            "topic": "Generative AI System Design",
            "suggested_role": "AI Architect"
        },
        {
            "id": "AI_07",
            "question": "What causes hallucinations in Large Language Models, and what engineering methods reduce them in production?",
            "topic": "LLM Reliability & Guardrails",
            "suggested_role": "AI Engineer"
        },
        {
            "id": "AI_08",
            "question": "What is the difference between Data Drift and Concept Drift, and how do you monitor for them in production ML pipelines?",
            "topic": "MLOps & Monitoring",
            "suggested_role": "MLOps Engineer"
        },
        {
            "id": "AI_09",
            "question": "What is target encoding, what risk does it pose, and how do you prevent target leakage when using it?",
            "topic": "Feature Engineering & Data Preprocessing",
            "suggested_role": "Data Scientist"
        },
        {
            "id": "AI_10",
            "question": "What is Chain-of-Thought (CoT) prompting, and how does Few-Shot learning improve LLM reasoning accuracy?",
            "topic": "Prompt Engineering & Reasoning",
            "suggested_role": "AI Engineer"
        }
    ],
    "hr_behavioural": [
        {
            "id": "HR_01",
            "question": "Tell me about a time you had a technical disagreement with a colleague or senior engineer. How did you resolve it?",
            "topic": "Conflict Resolution & Collaboration",
            "suggested_role": "All Engineering Roles"
        },
        {
            "id": "HR_02",
            "question": "Can you describe a significant mistake or failure you experienced at work or in a project, and what you learned from it?",
            "topic": "Accountability & Growth Mindset",
            "suggested_role": "All Engineering Roles"
        },
        {
            "id": "HR_03",
            "question": "How do you handle situations where you face multiple competing high-priority deadlines simultaneously?",
            "topic": "Prioritization & Pressure Management",
            "suggested_role": "All Engineering Roles"
        },
        {
            "id": "HR_04",
            "question": "Tell me about a time you had to lead a project or influence team members without having formal managerial authority.",
            "topic": "Informal Leadership & Influence",
            "suggested_role": "Senior / Lead Roles"
        },
        {
            "id": "HR_05",
            "question": "Describe a situation where project requirements or business priorities changed drastically midway through. How did you adapt?",
            "topic": "Adaptability & Agility",
            "suggested_role": "All Roles"
        },
        {
            "id": "HR_06",
            "question": "Tell me about a time you received critical or negative feedback from a manager or peer. How did you respond?",
            "topic": "Receptiveness to Feedback",
            "suggested_role": "All Roles"
        },
        {
            "id": "HR_07",
            "question": "Have you ever faced a situation where you were asked to cut corners or compromise quality or security? How did you respond?",
            "topic": "Professional Integrity & Ethics",
            "suggested_role": "All Roles"
        },
        {
            "id": "HR_08",
            "question": "How do you handle working alongside a teammate who is consistently unresponsive or failing to deliver on dependencies?",
            "topic": "Teamwork & De-escalation",
            "suggested_role": "All Roles"
        }
    ]
}


class InterviewSession:
    """Encapsulates the state and execution of an interactive interview session."""

    def __init__(
        self,
        role: str = "Software Engineer",
        experience_level: str = "Mid-Level",
        category: str = "software_development",
        total_questions: int = 4
    ):
        self.role = role
        self.experience_level = experience_level
        self.category = category if category in QUESTION_BANK else "software_development"
        self.total_questions = total_questions
        self.current_index = 0
        self.session_history: List[Dict[str, Any]] = []
        self.active_followup: Optional[str] = None
        self.is_completed = False

        # Prepare question queue
        candidates = list(QUESTION_BANK[self.category])
        random.seed(random.randint(1, 10000))
        random.shuffle(candidates)
        self.questions_queue = candidates[:total_questions]

    def get_current_question(self) -> Optional[str]:
        """Returns the active question string or None if finished."""
        if self.current_index < len(self.questions_queue):
            return self.questions_queue[self.current_index]["question"]
        return None

    def get_current_topic(self) -> str:
        """Returns the topic/competency of the active question."""
        if self.current_index < len(self.questions_queue):
            return self.questions_queue[self.current_index].get("topic", "General")
        return "Complete"

    def submit_answer(
        self,
        candidate_answer: str,
        model: Any,
        tokenizer: Any
    ) -> Dict[str, Any]:
        """
        Submits candidate answer, queries LLM with rubric prompt,
        validates output JSON, and updates session history.
        """
        current_q = self.get_current_question()
        if not current_q:
            raise ValueError("No active question to evaluate.")

        # Construct rubric evaluation prompt
        eval_prompt = build_evaluation_prompt(
            role=self.role,
            experience_level=self.experience_level,
            category=self.category,
            question=current_q,
            candidate_answer=candidate_answer,
            previous_followup=self.active_followup
        )

        # Call model
        raw_response = generate_model_response(
            model=model,
            tokenizer=tokenizer,
            prompt=eval_prompt,
            system_prompt=SYSTEM_INSTRUCTION,
            max_new_tokens=600,
            temperature=0.2
        )

        # Extract & sanitize JSON feedback
        parsed = extract_json_from_text(raw_response)
        feedback = sanitize_feedback(parsed, raw_response, current_q)

        # Store follow-up question
        self.active_followup = feedback.get("follow_up_question")

        # Record in history
        history_entry = {
            "question_index": self.current_index + 1,
            "question": current_q,
            "candidate_answer": candidate_answer,
            "feedback": feedback,
            "follow_up": self.active_followup,
            "raw_output": raw_response
        }
        self.session_history.append(history_entry)

        return feedback

    def advance_next_question(self) -> bool:
        """Advances to next question. Returns True if next question exists, False if session ends."""
        self.current_index += 1
        self.active_followup = None
        if self.current_index >= len(self.questions_queue):
            self.is_completed = True
            return False
        return True

    def finish_interview(self) -> Dict[str, Any]:
        """Finalizes the interview and produces the comprehensive performance report."""
        self.is_completed = True
        return generate_final_report(
            session_history=self.session_history,
            role=self.role,
            experience_level=self.experience_level,
            category=self.category
        )
