"""Run golden-dataset evaluations and record aggregate metrics in MLflow."""

import os
import json
import time
import re
import mlflow
import coach

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

def load_golden_dataset(dataset_version: str = "v1") -> list:
    """
    Dynamically loads the requested benchmark dataset version from the datasets folder.
    Example: dataset_version="v1" loads datasets/v1_golden_dataset.json
    """
    filename = f"{dataset_version}_golden_dataset.json"
    filepath = os.path.join(DATASETS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Golden dataset not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def contains_quantifiable_metrics(text: str) -> bool:
    """Checks if a rewritten bullet point contains numbers, percentages, or metrics."""
    return bool(re.search(r"\d+(\.\d+)?%?|\$\d+", text))

def run_evaluation(
    prompt_version: str = "v1", 
    dataset_version: str = "v1", 
    experiment_name: str = "Resume_Coach_Evals"
):
    """
    Executes automated evaluation across the requested Golden Dataset version
    and logs all parameters, metrics, and artifacts to MLflow.
    """
    dataset = load_golden_dataset(dataset_version=dataset_version)
    mlflow.set_experiment(experiment_name)

    print(f"\n🚀 Starting MLflow Evaluation Run:")
    print(f"   • Prompt Version:  [{prompt_version}]")
    print(f"   • Dataset Version: [{dataset_version}] ({len(dataset)} test cases)")
    print("-" * 65)

    with mlflow.start_run(run_name=f"eval_{prompt_version}_{dataset_version}_{int(time.time())}"):
        # 1. Log System Parameters to MLflow
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("model_name", "openai/gpt-oss-120b")
        mlflow.log_param("test_cases_count", len(dataset))

        total_latency = 0.0
        score_accuracy_hits = 0
        total_expected_missing_skills = 0
        caught_missing_skills = 0
        total_star_bullets = 0
        quantified_star_bullets = 0

        # 2. Iterate through each test case
        for idx, case in enumerate(dataset, 1):
            test_id = case["test_id"]
            print(f"[{idx}/{len(dataset)}] Running: {test_id} ({case['description']})...")

            start_time = time.time()
            try:
                result = coach.analyze_resume(
                    case["resume_text"], 
                    case["job_description"], 
                    prompt_version=prompt_version
                )
            except Exception as e:
                print(f"  ❌ Error executing test case: {e}")
                continue

            latency = round(time.time() - start_time, 2)
            total_latency += latency

            match_score = result.get("match_score", 0)
            missing_skills = result.get("missing_skills", [])
            star_rewrites = result.get("star_rewrites", [])

            # Check Metric 1: Score Range Accuracy
            min_score = case["expected_min_score"]
            max_score = case["expected_max_score"]
            in_range = min_score <= match_score <= max_score
            if in_range:
                score_accuracy_hits += 1

            # Check Metric 2: Missing Skills Recall
            expected_missing = case.get("expected_missing_skills", [])
            for exp_skill in expected_missing:
                total_expected_missing_skills += 1
                if any(exp_skill.lower() in found.lower() for found in missing_skills):
                    caught_missing_skills += 1

            # Check Metric 3: STAR Quantification Check
            for rewrite_item in star_rewrites:
                total_star_bullets += 1
                rewritten_text = rewrite_item.get("rewritten", "")
                if contains_quantifiable_metrics(rewritten_text):
                    quantified_star_bullets += 1

            print(f"  • Score: {match_score}% (Target: {min_score}-{max_score}%) -> {'✅ Pass' if in_range else '⚠️ Out of Range'}")
            print(f"  • Latency: {latency}s")


        # 3. Compute Aggregated Metrics
        avg_latency = round(total_latency / len(dataset), 2)
        score_accuracy_pct = round((score_accuracy_hits / len(dataset)) * 100, 1)
        skill_recall_pct = (
            round((caught_missing_skills / total_expected_missing_skills) * 100, 1)
            if total_expected_missing_skills > 0 else 100.0
        )
        star_quant_pct = (
            round((quantified_star_bullets / total_star_bullets) * 100, 1)
            if total_star_bullets > 0 else 100.0
        )


        # 4. Log Aggregated Metrics to MLflow
        mlflow.log_metric("avg_latency_seconds", avg_latency)
        mlflow.log_metric("score_accuracy_pct", score_accuracy_pct)
        mlflow.log_metric("missing_skills_recall_pct", skill_recall_pct)
        mlflow.log_metric("star_quantification_pct", star_quant_pct)

        print("\n" + "=" * 65)
        print("🏆 EVALUATION RUN COMPLETE - AGGREGATED METRICS:")
        print("=" * 65)
        print(f"• Score Range Accuracy:      {score_accuracy_pct}%")
        print(f"• Missing Skills Recall:      {skill_recall_pct}%")
        print(f"• STAR Quantification Rate:   {star_quant_pct}%")
        print(f"• Average Latency:            {avg_latency}s")
        print("=" * 65)
        print("All metrics logged to MLflow successfully!\n")

        return {
            "score_accuracy_pct": score_accuracy_pct,
            "missing_skills_recall_pct": skill_recall_pct,
            "star_quantification_pct": star_quant_pct,
            "avg_latency_seconds": avg_latency,
            "test_cases_count": len(dataset)
        }

if __name__ == "__main__":
    run_evaluation(prompt_version="v1", dataset_version="v1")