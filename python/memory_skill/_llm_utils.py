"""Shared LLM utilities — call DeepSeek API and parse JSON responses.

Used by TreeManager, KnowledgeSynth, and LearningDecider.
Single source of truth for API calls and JSON extraction.
"""

from __future__ import annotations

import json
import logging
import re
import time as _time

import openai

_logger = logging.getLogger(__name__)


def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 30.0,
    retries: int = 1,
) -> str | None:
    """Call the LLM API and return the raw response text, or None on failure.

    Retries up to *retries* times on empty response or service errors.
    """
    for attempt in range(retries + 1):
        try:
            client = openai.OpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=float(timeout),
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            content = msg.content
            # Reasoning models (e.g. deepseek-v4-flash) return the answer in
            # reasoning_content with empty content — fall back to it.
            if not content or not content.strip():
                content = getattr(msg, "reasoning_content", None)
            if content and content.strip():
                return content
        except Exception as exc:
            _logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries + 1, exc)
        if attempt < retries:
            _time.sleep(1.0 * (attempt + 1))
    return None


def parse_json_response(raw: str) -> dict | list | None:
    """Extract JSON from an LLM response (handles markdown fences)."""
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        else:
            text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try to extract the first JSON object/array from mixed text
    for pattern in [r"\[.*\]", r"\{.*\}"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    _logger.warning("Failed to parse JSON from LLM response: %.100s", raw)
    return None
