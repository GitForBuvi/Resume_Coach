import os
import re
from datetime import date
from typing import Tuple, List

# In-memory quota store: { user_identifier: {"date": date_obj, "count": int} }
_USER_QUOTA_STORE = {}

PATTERNS_FILE_PATH = os.path.join(os.path.dirname(__file__), "injection_patterns.txt")

def load_injection_patterns(filepath: str = PATTERNS_FILE_PATH) -> List[str]:
    """
    Dynamically loads injection patterns from an external file.
    Ignores comments (#) and empty lines.
    """
    patterns = []
    if not os.path.exists(filepath):
        return patterns

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.strip()
            # Ignore blank lines and comments
            if cleaned_line and not cleaned_line.startswith("#"):
                patterns.append(cleaned_line)
    return patterns

def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    """
    Scans input text against dynamically loaded injection patterns.
    Returns:
        (is_suspicious: bool, reason: str)
    """
    cleaned_lower = text.lower()
    patterns = load_injection_patterns()

    for pattern in patterns:
        match = re.search(pattern, cleaned_lower)
        if match:
            matched_phrase = match.group(0)
            return True, f"Suspicious instruction detected in document: '{matched_phrase}'"

    return False, "Clean"

def sanitize_and_fence(text: str, tag_name: str = "untrusted_candidate_resume") -> str:
    """
    Fences untrusted input within XML boundary tags.
    Prevents tag-escape attacks by neutralizing closing tags.
    """
    sanitized_text = re.sub(rf"</?\s*{tag_name}\s*>", "", text, flags=re.IGNORECASE)
    return f"<{tag_name}>\n{sanitized_text.strip()}\n</{tag_name}>"

def check_and_increment_quota(user_id: str = "default_user", max_daily_limit: int = 5) -> Tuple[bool, int, int]:
    """
    Tracks and enforces daily usage quotas per user.
    Returns:
        (is_allowed: bool, current_usage: int, max_limit: int)
    """
    today = date.today()
    record = _USER_QUOTA_STORE.get(user_id)

    if not record or record["date"] != today:
        _USER_QUOTA_STORE[user_id] = {"date": today, "count": 1}
        return True, 1, max_daily_limit

    if record["count"] >= max_daily_limit:
        return False, record["count"], max_daily_limit

    record["count"] += 1
    return True, record["count"], max_daily_limit