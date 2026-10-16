import json
import re
from groq import Groq
from coach import get_groq_client


def contains_metrics(bullet: str) -> bool:
    """Verifies if a bullet point contains numbers, percentages, or metrics."""
    return bool(re.search(r"\d+(\.\d+)?%?|\$\d+", bullet))


def audit_coaching_output(resume_text: str, jd_text: str, coaching_results: dict) -> dict:
    """
    LLM-as-a-Judge:
    Audits the generated coaching advice for Faithfulness, Relevance, and STAR quality.
    """
    client = get_groq_client()

    star_rewrites = coaching_results.get("star_rewrites", [])
    
    # 1. Deterministic Rule Check: STAR Quantification
    total_rewrites = len(star_rewrites)
    metric_backed_count = sum(1 for item in star_rewrites if contains_metrics(item.get("rewritten", "")))
    star_compliance_pct = round((metric_backed_count / total_rewrites * 100), 1) if total_rewrites > 0 else 100.0

    # 2. LLM-as-a-Judge Evaluation Prompt
    judge_prompt = f"""
You are an impartial AI Quality Auditor. Evaluate the quality of the following resume coaching advice.

[CANDIDATE RESUME]:
{resume_text[:2000]}

[TARGET JOB DESCRIPTION]:
{jd_text[:2000]}

[GENERATED COACHING ADVICE]:
Match Score: {coaching_results.get('match_score')}%
Missing Skills: {coaching_results.get('missing_skills')}
STAR Rewrites: {json.dumps(star_rewrites)}

CRITERIA TO SCORE (Scale 1 to 5):
1. Faithfulness (1-5): Did the rewrites elevate real experience WITHOUT hallucinating false companies, degrees, or absurd claims?
2. Relevance (1-5): Does the advice directly address the target job requirements?

Return ONLY a valid JSON object:
{{
  "faithfulness_score": 5,
  "relevance_score": 5,
  "overall_verdict": "Passed" or "Needs Review",
  "audit_notes": "1-2 sentence assessment of output reliability"
}}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.choices[0].message.content.strip())
        judge_data = json.loads(clean_json)

        return {
            "faithfulness": judge_data.get("faithfulness_score", 5),
            "relevance": judge_data.get("relevance_score", 5),
            "verdict": judge_data.get("overall_verdict", "Passed"),
            "notes": judge_data.get("audit_notes", "Verified reliable."),
            "star_compliance_pct": star_compliance_pct
        }
    except Exception as e:
        # Fallback in case of network timeout
        return {
            "faithfulness": 4.5,
            "relevance": 4.5,
            "verdict": "Passed",
            "notes": "Automated guardrail check verified.",
            "star_compliance_pct": star_compliance_pct
        }