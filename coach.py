"""Resume-to-job-description analysis using versioned prompts and the Groq API."""

import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
import security

# 1. Load the secret API key from the .env file
load_dotenv()

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def get_groq_client() -> Groq:
    """Initializes and returns the Groq client securely."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
    return Groq(api_key=api_key)

def load_system_prompt(version: str = "v1") -> str:
    """
    Dynamically loads the system prompt from the prompts directory.
    Example: version="v1" loads prompts/v1_system_coach.txt
    """
    filename = f"{version}_system_coach.txt"
    filepath = os.path.join(PROMPTS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt template not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def analyze_resume(resume_text: str, job_description: str, prompt_version: str = "v1") -> dict:
    """
    Analyzes resume text against a target job description.
    Includes pre-flight prompt injection checks and XML boundary fencing.
    """
    # 🛡️ Guardrail 1: Pre-flight Injection Scan
    is_suspicious, reason = security.detect_prompt_injection(resume_text)
    if is_suspicious:
        raise ValueError(f"Security Alert: {reason}. Analysis halted.")

    client = get_groq_client()

    # 1. Load the requested prompt version
    system_prompt = load_system_prompt(version=prompt_version)

    # 🛡️ Guardrail 2: XML Boundary Fencing for untrusted candidate input
    fenced_resume = security.sanitize_and_fence(resume_text, tag_name="untrusted_candidate_resume")

    user_prompt = f"""
--- TARGET JOB DESCRIPTION ---
{job_description}

--- CANDIDATE RESUME ---
{fenced_resume}
"""


    # 2. Call Groq API with JSON mode
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw_content = response.choices[0].message.content

    # 3. Defensive JSON Parsing
    clean_json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip())

    try:
        parsed_data = json.loads(clean_json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse AI response as JSON: {e}\nRaw output: {raw_content}")