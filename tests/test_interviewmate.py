"""
InterviewMate Unit & Integration Test Suite
===========================================
Validates:
1. Dataset generation, schema integrity, and zero data leakage.
2. Hardware check and ModelLoader fallbacks.
3. JSON extraction, sanitization, and repair mechanisms.
4. InterviewEngine session lifecycle, question progression, and follow-ups.
5. Final performance report generation and 7-day study plan.
6. Comparative evaluation benchmarking.
"""

import json
import os
import sys
import unittest

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feedback import (
    extract_json_from_text,
    sanitize_feedback,
    build_evaluation_prompt,
    generate_final_report
)
from src.interview_engine import InterviewSession, QUESTION_BANK
from src.model_loader import check_gpu_status, load_model_and_tokenizer
from src.evaluation import evaluate_single_sample, run_evaluation_benchmark, compare_models


class TestDatasetIntegrity(unittest.TestCase):
    """Tests the generated dataset files and verifies lack of data leakage."""

    def setUp(self):
        self.train_path = os.path.join(BASE_DIR, "data", "interview_train.jsonl")
        self.eval_path = os.path.join(BASE_DIR, "data", "interview_eval.jsonl")

    def test_dataset_files_exist(self):
        self.assertTrue(os.path.exists(self.train_path), "interview_train.jsonl does not exist")
        self.assertTrue(os.path.exists(self.eval_path), "interview_eval.jsonl does not exist")

    def test_schema_and_counts(self):
        def read_records(path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        train_records = read_records(self.train_path)
        eval_records = read_records(self.eval_path)
        total_records = len(train_records) + len(eval_records)

        self.assertGreaterEqual(total_records, 100, "Dataset should have at least 100 records")
        self.assertLessEqual(total_records, 200, "Dataset should have at most 200 records")

        required_keys = [
            "id", "category", "role", "experience_level", "instruction",
            "interview_question", "candidate_answer", "answer_quality",
            "scoring_rubric", "overall_score", "strengths", "weaknesses",
            "actionable_feedback", "improved_sample_answer", "follow_up_question"
        ]

        for r in train_records + eval_records:
            for k in required_keys:
                self.assertIn(k, r, f"Record {r.get('id')} missing key '{k}'")
            self.assertGreaterEqual(r["overall_score"], 1)
            self.assertLessEqual(r["overall_score"], 10)
            self.assertIn(r["answer_quality"], ["weak", "average", "strong"])

    def test_zero_data_leakage(self):
        """Verifies no scenario overlap between training and evaluation splits."""
        with open(self.train_path, "r", encoding="utf-8") as f:
            train_scenarios = {json.loads(l)["id"].rsplit("_", 1)[0] for l in f if l.strip()}
        with open(self.eval_path, "r", encoding="utf-8") as f:
            eval_scenarios = {json.loads(l)["id"].rsplit("_", 1)[0] for l in f if l.strip()}

        overlap = train_scenarios.intersection(eval_scenarios)
        self.assertEqual(len(overlap), 0, f"Data leakage detected! Shared scenarios: {overlap}")


class TestModelLoaderAndHardware(unittest.TestCase):
    """Tests hardware detection and fallback loading."""

    def test_hardware_status_dict(self):
        status = check_gpu_status()
        self.assertIn("hf_available", status)
        self.assertIn("cuda_available", status)
        self.assertIn("device_name", status)
        self.assertIn("total_memory_gb", status)
        self.assertIn("free_memory_gb", status)

    def test_model_loading_fallback(self):
        model, tokenizer, is_adapter = load_model_and_tokenizer(load_in_4bit=False)
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)


class TestFeedbackAndJSONParser(unittest.TestCase):
    """Tests JSON extraction, regex repair, rubric bounds, and fallback heuristics."""

    def test_clean_json_parsing(self):
        raw = json.dumps({
            "overall_score": 8,
            "scoring_rubric": {"relevance": 8, "accuracy_depth": 8, "clarity_structure": 8, "completeness": 8},
            "strengths": ["Clear explanation"],
            "weaknesses": ["Could add numbers"],
            "actionable_feedback": ["Quantify metrics"],
            "improved_sample_answer": "An ideal answer.",
            "follow_up_question": "How to scale?"
        })
        parsed = extract_json_from_text(raw)
        self.assertIsNotNone(parsed)
        feedback = sanitize_feedback(parsed, raw, "Test Question")
        self.assertEqual(feedback["overall_score"], 8.0)
        self.assertFalse(feedback["is_fallback"])

    def test_markdown_code_fence_parsing(self):
        raw = """Here is your evaluation:
```json
{
  "overall_score": 9.5,
  "scoring_rubric": {"relevance": 10, "accuracy_depth": 9, "clarity_structure": 9, "completeness": 10},
  "strengths": ["Brilliant depth"],
  "weaknesses": ["Minor edge case missing"],
  "actionable_feedback": ["Review error handling"],
  "improved_sample_answer": "Ideal response text.",
  "follow_up_question": "What about partitions?"
}
```
Hope this helps!"""
        parsed = extract_json_from_text(raw)
        self.assertIsNotNone(parsed)
        feedback = sanitize_feedback(parsed, raw, "Test Question")
        self.assertEqual(feedback["overall_score"], 9.5)

    def test_trailing_comma_repair(self):
        raw = """{
            "overall_score": 7,
            "scoring_rubric": {
                "relevance": 7,
                "accuracy_depth": 7,
                "clarity_structure": 7,
                "completeness": 7,
            },
            "strengths": ["Good basic concept",],
            "weaknesses": ["Misses trade-offs",],
            "actionable_feedback": ["Add detail",],
            "improved_sample_answer": "Better version",
            "follow_up_question": "Next step?",
        }"""
        parsed = extract_json_from_text(raw)
        self.assertIsNotNone(parsed)
        feedback = sanitize_feedback(parsed, raw, "Test Question")
        self.assertEqual(feedback["overall_score"], 7.0)

    def test_broken_output_fallback_never_crashes(self):
        raw_garbage = "I am an AI and I think this answer is mediocre. overall_score: 5. Good attempt."
        parsed = extract_json_from_text(raw_garbage)
        self.assertIsNone(parsed)
        feedback = sanitize_feedback(parsed, raw_garbage, "Test Question")
        self.assertEqual(feedback["overall_score"], 5.0)
        self.assertTrue(feedback["is_fallback"])
        self.assertIn("strengths", feedback)
        self.assertIn("weaknesses", feedback)
        self.assertIn("actionable_feedback", feedback)

    def test_score_boundary_clamping(self):
        # Out-of-bounds score (e.g. 15 or -5) should be clamped strictly between 1 and 10
        raw = json.dumps({"overall_score": 99, "scoring_rubric": {"relevance": 50}})
        parsed = extract_json_from_text(raw)
        feedback = sanitize_feedback(parsed, raw, "Test Question")
        self.assertEqual(feedback["overall_score"], 10.0)
        self.assertEqual(feedback["scoring_rubric"]["relevance"], 10)


class TestInterviewEngine(unittest.TestCase):
    """Tests interview session flow, follow-ups, and report generation."""

    def test_session_lifecycle(self):
        model, tokenizer, _ = load_model_and_tokenizer(load_in_4bit=False)
        session = InterviewSession(
            role="Backend Engineer",
            experience_level="Mid-Level",
            category="software_development",
            total_questions=2
        )

        # 1. Check initial state
        q1 = session.get_current_question()
        self.assertIsNotNone(q1)
        self.assertEqual(session.current_index, 0)

        # 2. Submit answer
        fb1 = session.submit_answer(
            candidate_answer="Indexes use B-Trees to provide O(log N) lookups, but slow down writes.",
            model=model,
            tokenizer=tokenizer
        )
        self.assertIn("overall_score", fb1)
        self.assertIn("scoring_rubric", fb1)
        self.assertIn("improved_sample_answer", fb1)
        self.assertIn("follow_up_question", fb1)

        # 3. Advance to question 2
        has_next = session.advance_next_question()
        self.assertTrue(has_next)
        self.assertEqual(session.current_index, 1)

        # 4. Submit answer 2
        fb2 = session.submit_answer(
            candidate_answer="REST uses HTTP verbs like GET and POST while GraphQL uses a single endpoint.",
            model=model,
            tokenizer=tokenizer
        )
        self.assertIn("overall_score", fb2)

        # 5. Advance past end
        has_next = session.advance_next_question()
        self.assertFalse(has_next)
        self.assertTrue(session.is_completed)

        # 6. Generate final report
        report = session.finish_interview()
        self.assertIn("average_score", report)
        self.assertIn("grade", report)
        self.assertIn("study_plan", report)
        self.assertIn("markdown_report", report)
        self.assertEqual(len(report["study_plan"]), 5)


class TestEvaluationEngine(unittest.TestCase):
    """Tests the comparative evaluation metrics."""

    def test_evaluation_benchmark(self):
        model, tokenizer, _ = load_model_and_tokenizer(load_in_4bit=False)
        benchmark = run_evaluation_benchmark(model, tokenizer, "Test Model", max_samples=2)

        self.assertEqual(benchmark["samples_evaluated"], 2)
        self.assertIn("json_compliance_pct", benchmark)
        self.assertIn("score_mae", benchmark)
        self.assertIn("completeness_pct", benchmark)

        # Comparison table generator
        comp_md = compare_models(benchmark, benchmark)
        self.assertIn("Model Evaluation Comparison", comp_md)
        self.assertIn("JSON Schema Compliance", comp_md)


if __name__ == "__main__":
    unittest.main()
