"""Tree Classifier — LLM-based content-to-branch routing.

Separated from TreeManager so the tree remains a pure index.
Uses ``_llm_utils.call_llm`` for API calls.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

_TREE_CLASSIFY_PROMPT = """\
Content: "{content}"

Classify into one branch. Return JSON only:
{{"root": "user"|"assistant", "branch": "pref"|"mem"|"pers"|"task"|"skill"}}

- user/pref: preferences, habits, likes/dislikes, routines
- user/mem: memories, life events, casual updates, observations
- assistant/pers: personality traits, tone, style, behavior rules
- assistant/task: specific projects in progress, bugs to fix, deadlines, todos
- assistant/skill: reusable knowledge, tools mastered, techniques learned, APIs, frameworks
"""

_NAVIGATE_PROMPT = """\
Available branches:
  user_pref (偏好): habits, likes, dislikes, routines
  user_mem (回忆与目标): life events, meals, mood, casual updates
  assistant_task (任务): specific projects, bugs, deadlines, todos
  assistant_skill (技能): tools mastered, APIs, frameworks, techniques
  assistant_pers (人格): personality traits (rarely used)

User: "{query}"

Pick branch(es) + days to search. JSON only:
{{"searches": [{{"branch": "user_mem", "days": 3}}]}}

Rules:
- "习惯"/"讨厌"/"喜欢"/"总是" → check user_pref
- "项目"/"bug"/"todo"/"进展" → check assistant_task
- "学"/"API"/"框架"/"怎么用" → check assistant_skill
- Default → user_mem last 3 days
"""


class TreeClassifier:
    """LLM-based content classifier that determines which tree branch a
    piece of content belongs to.

    Parameters
    ----------
    api_base: DeepSeek API base URL.
    api_key: DeepSeek API key.
    model: Model name (e.g. ``deepseek-v4-flash``).
    """

    def __init__(self, api_base: str, api_key: str, model: str):
        self._api_base = api_base
        self._api_key = api_key
        self._model = model

    def classify(self, content: str) -> dict:
        """Return ``{"root": ..., "branch": ...}`` for *content*.

        On API failure, returns ``{"root": "user", "branch": "mem"}``.
        """
        prompt = _TREE_CLASSIFY_PROMPT.format(content=content[:500])
        try:
            result = _call_llm(
                self._api_base, self._api_key, self._model, prompt,
            )
            if result and "branch" in result:
                _logger.debug("Tree classify: %s → %s", content[:60], result)
                return result
        except Exception as exc:
            _logger.warning("Tree classify API failed: %s, using fallback", exc)
        return _fallback_classify()

    def navigate(self, query: str, max_tokens: int = 512) -> list[dict] | None:
        """Ask LLM to select branches + time ranges for *query*.

        Returns a list of ``{"branch": str, "days": int}`` dicts,
        or ``None`` on any failure (caller falls back to keyword search).
        """
        prompt = _NAVIGATE_PROMPT.format(query=query[:500])
        try:
            result = _call_llm(
                self._api_base, self._api_key, self._model, prompt,
                max_tokens=max_tokens,
            )
            if result and "searches" in result:
                searches = result["searches"]
                if isinstance(searches, list) and len(searches) > 0:
                    valid: list[dict] = []
                    for s in searches:
                        if isinstance(s, dict) and "branch" in s:
                            valid.append({
                                "branch": s["branch"],
                                "days": int(s.get("days", 3)),
                            })
                    if valid:
                        _logger.debug("Navigate LLM selected: %s", valid)
                        return valid
        except Exception as exc:
            _logger.warning("Navigate LLM failed: %s", exc)
        return None


def _call_llm(api_base: str, api_key: str, model: str, prompt: str,
              max_tokens: int = 256) -> dict | None:
    from memory_skill._llm_utils import call_llm, parse_json_response
    import logging
    raw = call_llm(api_base, api_key, model, prompt,
                   max_tokens=max_tokens, temperature=0.0)
    if raw:
        result = parse_json_response(raw)
        return result if isinstance(result, dict) else None
    logging.getLogger(__name__).debug("LLM returned empty for prompt: %.100s", prompt)
    return None


def _fallback_classify() -> dict:
    return {"root": "user", "branch": "mem"}


def classify_skill_path(title: str) -> list[str]:
    """Derive a tree path from a skill title (first words as hierarchy)."""
    parts = title.strip().split()
    if not parts:
        return []
    first = parts[0]
    rest = " ".join(parts[1:]) if len(parts) > 1 else None
    path = [first]
    if rest:
        path.append(rest)
    return path
