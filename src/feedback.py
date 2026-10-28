"""
InterviewMate Feedback Engine & Rubric Parser
=============================================
Manages prompt construction, robust JSON extraction and repair,
rubric scoring validation, and final performance report generation.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple


SYSTEM_INSTRUCTION = (
    "You are InterviewMate, an expert AI mock interviewer and senior hiring committee member. "
    "Your mission is to evaluate the candidate's interview response with technical precision, "
    "fairness, and constructive coaching. "
    "You MUST respond ONLY with a single valid JSON object containing no markdown outside the JSON. "
    "Follow the requested schema strictly."
)

RUBRIC_GUIDELINES = {
    "relevance": "How directly and accurately the candidate answered the specific question asked.",
    "accuracy_depth": "Technical correctness, architectural depth, complexity awareness, and absence of misconceptions.",
    "clarity_structure": "Coherence, communication quality, structured explanation (e.g. STAR method, trade-offs).",
    "completeness": "Whether key edge cases, trade-offs, scalability, or failure modes were addressed."
}


def build_evaluation_prompt(
    role: str,
    experience_level: str,
    category: str,
    question: str,
    candidate_answer: str,
    previous_followup: Optional[str] = None
) -> str:
    """Constructs the prompt for the model to evaluate a candidate's answer."""
    context_str = ""
    if previous_followup:
        context_str = f"Context (Previous Follow-up Question): {previous_followup}\n"

    prompt = f"""
Candidate Target Role: {role}
Experience Level: {experience_level}
Interview Track: {category}
{context_str}
Interview Question:
"{question}"

Candidate's Answer:
"{candidate_answer}"

Scoring Rubric (Evaluate each metric from 1 to 10):
1. relevance (1-10): {RUBRIC_GUIDELINES['relevance']}
2. accuracy_depth (1-10): {RUBRIC_GUIDELINES['accuracy_depth']}
3. clarity_structure (1-10): {RUBRIC_GUIDELINES['clarity_structure']}
4. completeness (1-10): {RUBRIC_GUIDELINES['completeness']}

Provide your evaluation strictly as a valid JSON object matching this schema:
{{
  "overall_score": <integer or float from 1 to 10>,
  "scoring_rubric": {{
    "relevance": <integer 1-10>,
    "accuracy_depth": <integer 1-10>,
    "clarity_structure": <integer 1-10>,
    "completeness": <integer 1-10>
  }},
  "strengths": [
    "<specific strong point in the answer>"
  ],
  "weaknesses": [
    "<specific gap, flaw, or missing detail>"
  ],
  "actionable_feedback": [
    "<concrete recommendation to improve this answer>"
  ],
  "improved_sample_answer": "<an articulate, model answer demonstrating high competence>",
  "follow_up_question": "<a natural, probing follow-up question based directly on their answer>"
}}
"""
    return prompt.strip()


def extract_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    """Attempts multiple extraction strategies to parse JSON from model output."""
    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # Strategy 1: Direct JSON parsing
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Strategy 2: Extract from markdown code fences ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(fence_pattern, cleaned, re.IGNORECASE)
    for block in matches:
        try:
            return json.loads(block.strip())
        except Exception:
            pass

    # Strategy 3: Find outermost curly braces { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_slice = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(json_slice)
        except Exception:
            # Try repairing common minor JSON issues (trailing commas before } or ])
            repaired = re.sub(r",\s*([\]}])", r"\1", json_slice)
            try:
                return json.loads(repaired)
            except Exception:
                pass

    return None


def sanitize_feedback(parsed: Optional[Dict[str, Any]], raw_text: str, question: str) -> Dict[str, Any]:
    """
    Guarantees that all required feedback fields exist and conform to types/ranges.
    Provides rule-based fallback if parsing fails completely so the application NEVER crashes.
    """
    if parsed is None:
        # Fallback heuristic parser
        score_match = re.search(r"overall_score[\"':\s]+([0-9]+(?:\.[0-9]+)?)", raw_text)
        fallback_score = float(score_match.group(1)) if score_match else 6.0
        fallback_score = max(1.0, min(10.0, fallback_score))

        return {
            "overall_score": round(fallback_score, 1),
            "scoring_rubric": {
                "relevance": int(fallback_score),
                "accuracy_depth": int(fallback_score),
                "clarity_structure": int(fallback_score),
                "completeness": int(fallback_score)
            },
            "strengths": ["Demonstrated basic familiarity with the topic."],
            "weaknesses": ["Answer could be more structured and include concrete architectural trade-offs."],
            "actionable_feedback": [
                "Structure explanations using cause-and-effect or the STAR method.",
                "Mention scalability, error handling, and concrete operational metrics."
            ],
            "improved_sample_answer": f"A comprehensive response to '{question}' clearly defines core terms, analyzes trade-offs, and provides concrete examples.",
            "follow_up_question": "Can you provide a specific production example where this trade-off affected system design?",
            "is_fallback": True
        }

    # Validate and clamp score
    overall = parsed.get("overall_score", 6.0)
    try:
        overall = float(overall)
    except (ValueError, TypeError):
        overall = 6.0
    overall = round(max(1.0, min(10.0, overall)), 1)

    # Validate rubric
    rubric = parsed.get("scoring_rubric", {})
    sanitized_rubric = {}
    for k in ["relevance", "accuracy_depth", "clarity_structure", "completeness"]:
        val = rubric.get(k, int(overall))
        try:
            val = int(val)
        except (ValueError, TypeError):
            val = int(overall)
        sanitized_rubric[k] = max(1, min(10, val))

    # Validate lists
    def clean_list(items: Any, default_item: str) -> List[str]:
        if isinstance(items, list) and items:
            cleaned = [str(x).strip() for x in items if str(x).strip()]
            return cleaned if cleaned else [default_item]
        elif isinstance(items, str) and items.strip():
            return [items.strip()]
        return [default_item]

    strengths = clean_list(parsed.get("strengths"), "Attempted the question directly.")
    weaknesses = clean_list(parsed.get("weaknesses"), "Could delve deeper into operational trade-offs.")
    advice = clean_list(parsed.get("actionable_feedback"), "Focus on structured reasoning and quantifiable metrics.")

    improved = parsed.get("improved_sample_answer", "")
    if not isinstance(improved, str) or not improved.strip():
        improved = f"An optimal response to '{question}' clearly articulates underlying mechanisms and practical trade-offs."

    follow_up = parsed.get("follow_up_question", "")
    if not isinstance(follow_up, str) or not follow_up.strip():
        follow_up = "How would you handle failure recovery or scaling in that scenario?"

    return {
        "overall_score": overall,
        "scoring_rubric": sanitized_rubric,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "actionable_feedback": advice,
        "improved_sample_answer": improved.strip(),
        "follow_up_question": follow_up.strip(),
        "is_fallback": False
    }


def generate_final_report(
    session_history: List[Dict[str, Any]],
    role: str,
    experience_level: str,
    category: str
) -> Dict[str, Any]:
    """
    Compiles full interview transcript and individual scores into a comprehensive
    performance audit, grading tier, and actionable 7-day preparation plan.
    """
    if not session_history:
        return {
            "average_score": 0.0,
            "grade": "Incomplete",
            "summary": "No questions were completed in this session.",
            "rubric_averages": {},
            "study_plan": [],
            "markdown_report": "No session history recorded."
        }

    total_score = sum(entry["feedback"]["overall_score"] for entry in session_history)
    avg_score = round(total_score / len(session_history), 2)

    # Calculate rubric category averages
    rubric_totals = {"relevance": 0.0, "accuracy_depth": 0.0, "clarity_structure": 0.0, "completeness": 0.0}
    for entry in session_history:
        r = entry["feedback"]["scoring_rubric"]
        for k in rubric_totals:
            rubric_totals[k] += r.get(k, 6.0)

    num_q = len(session_history)
    rubric_averages = {k: round(v / num_q, 1) for k, v in rubric_totals.items()}

    # Determine Performance Tier
    if avg_score >= 8.5:
        grade = "Strong Hire (Offer Ready)"
        verdict = "Exceptional performance! Demonstrates deep domain expertise, clear articulation, and mastery of trade-offs."
    elif avg_score >= 7.0:
        grade = "Hire / Advancing to Next Round"
        verdict = "Solid performance. Meets professional expectations with good foundational knowledge and structure."
    elif avg_score >= 5.0:
        grade = "Borderline / Needs Polish"
        verdict = "Shows promise, but lacks technical depth, quantitative metrics, or structured delivery (e.g. STAR method)."
    else:
        grade = "Needs Fundamentals Review"
        verdict = "Needs significant preparation on fundamental concepts, accurate definitions, and structured communication."

    # Aggregate strengths & priority improvement areas
    all_strengths = []
    all_weaknesses = []
    all_advice = []
    for entry in session_history:
        all_strengths.extend(entry["feedback"]["strengths"])
        all_weaknesses.extend(entry["feedback"]["weaknesses"])
        all_advice.extend(entry["feedback"]["actionable_feedback"])

    # Unique selection
    top_strengths = list(dict.fromkeys(all_strengths))[:4]
    top_weaknesses = list(dict.fromkeys(all_weaknesses))[:4]
    top_advice = list(dict.fromkeys(all_advice))[:4]

    # Generate 7-Day Action Plan
    study_plan = [
        "Day 1-2 (Fundamentals): Revisit core definitions, time/space complexities, and underlying data structures.",
        "Day 3-4 (Frameworks & Depth): Practice the STAR method (Situation, Task, Action, Result) for behavioral questions and trade-off matrices for technical questions.",
        "Day 5 (Failure Modes & Scaling): Study how systems behave under stress, network partitions, high concurrency, and data anomalies.",
        "Day 6 (Mock Drills): Conduct timed 30-minute mock interviews speaking answers aloud without pausing.",
        "Day 7 (Review & Rest): Review your question log and feedback points, focusing on calm, structured delivery."
    ]

    # Build Markdown Report
    md_lines = [
        f"# 🎓 InterviewMate Final Performance Report",
        f"**Target Role:** {role} | **Level:** {experience_level} | **Mode:** {category.capitalize()}",
        f"**Questions Completed:** {num_q} | **Overall Average Score:** **{avg_score} / 10**",
        f"**Hiring Recommendation:** `{grade}`",
        f"\n> **Executive Summary:** {verdict}\n",
        f"### 📊 Rubric Breakdown",
        f"- **Relevance:** `{rubric_averages['relevance']} / 10`",
        f"- **Accuracy & Technical Depth:** `{rubric_averages['accuracy_depth']} / 10`",
        f"- **Clarity & Structure:** `{rubric_averages['clarity_structure']} / 10`",
        f"- **Completeness:** `{rubric_averages['completeness']} / 10`\n",
        f"### 🌟 Key Strengths",
    ]
    for s in top_strengths:
        md_lines.append(f"- {s}")

    md_lines.append(f"\n### ⚠️ Priority Growth Areas")
    for w in top_weaknesses:
        md_lines.append(f"- {w}")

    md_lines.append(f"\n### 💡 Recommended 7-Day Study Plan")
    for sp in study_plan:
        md_lines.append(f"- {sp}")

    md_lines.append(f"\n---\n### 📝 Detailed Session Log")
    for idx, item in enumerate(session_history, start=1):
        fb = item["feedback"]
        md_lines.append(f"\n#### Question {idx}: {item['question']}")
        md_lines.append(f"**Your Answer:** {item['candidate_answer']}")
        md_lines.append(f"**Score:** `{fb['overall_score']} / 10`")
        md_lines.append(f"**Improved Model Answer:** *{fb['improved_sample_answer']}*")
        if item.get("follow_up"):
            md_lines.append(f"**Follow-Up Generated:** *{item['follow_up']}*")

    markdown_report = "\n".join(md_lines)

    return {
        "average_score": avg_score,
        "grade": grade,
        "verdict": verdict,
        "rubric_averages": rubric_averages,
        "top_strengths": top_strengths,
        "top_weaknesses": top_weaknesses,
        "top_advice": top_advice,
        "study_plan": study_plan,
        "markdown_report": markdown_report
    }
