"""
Memory Skill — LLM-based Importance Gate (Phase 6.2).

Drop-in replacement for ``ImportanceScorer`` that uses an LLM (DeepSeek V4
Flash via OpenAI-compatible API) to classify message importance.  Falls back
to the rule-based scorer on any API failure.

Architecture::

    LLMImportanceGate.evaluate(content)
        ├── _classify_via_api(content, model)   ← DeepSeek V4 Flash
        └── _classify_via_rules(content)        ← ImportanceScorer (fallback)

Three-tier classification::

    trivial   (0.05) → greetings, small talk, fillers
    important (0.5)  → facts, preferences, decisions
    critical  (0.85) → explicit memory commands, strong preferences

Usage::

    from memory_skill.importance_llm import LLMImportanceGate

    gate = LLMImportanceGate()
    score, persist, category = gate.evaluate("你好")
    # → (0.05, False, "trivial")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

# ── Logger ──────────────────────────────────────────────────────────────────────

_log = logging.getLogger("memory_skill.importance_llm")

# ── Category score mapping ─────────────────────────────────────────────────────

_CATEGORY_SCORES: dict[str, float] = {
    "trivial": 0.05,
    "important": 0.5,
    "critical": 0.85,
}

_VALID_LABELS: set[str] = {"trivial", "important", "critical"}

# ── Default env values ──────────────────────────────────────────────────────────

_DEFAULT_API_BASE = "https://api.deepseek.com/v1"
_DEFAULT_API_KEY = "sk-no-key"
_DEFAULT_MODEL = "deepseek-v4-flash"
_DEFAULT_THRESHOLD = "0.3"
_DEFAULT_TIMEOUT = "5"

# ── Classification prompt ───────────────────────────────────────────────────────

_CLASSIFICATION_TEMPLATE = """\
Decide if this message should be saved to long-term memory. Answer with exactly one word — trivial, important, or critical.

trivial = casual chat, greetings, short acknowledgments ("ok", "好的", "嗯")
important = factual info, preferences, bug reports, task descriptions
critical = explicit requests to remember ("记住", "别忘了"), strong dislikes ("我讨厌"), urgent warnings

Message: {content}
Importance:"""

# DeepSeek V4 Flash ignores system messages entirely — classification
# instructions MUST be placed in the user message. The template above
# embeds the role inline so the model sees it as a natural task.
#
# max_tokens=1024: V4 Flash reasoning tokens are highly variable (30-150).
# This is far more than needed for a one-word response, but ensures
# reasoning never starves the visible output.
# internally before producing visible output. Values ≤20 cause
# the model to return empty responses (all tokens consumed by reasoning).

# ── Module-level lazy-init state ────────────────────────────────────────────────

_env_loaded: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _mask_key(key: str) -> str:
    """Return a masked version of an API key: ``sk-...XXXX``."""
    if not key or len(key) <= 8:
        return "sk-...(masked)"
    return f"{key[:4]}...{key[-4:]}"


def _load_env() -> None:
    """Load configuration from project-root ``.env`` file (idempotent).

    If ``.env`` is missing, logs a warning but does not crash — env vars
    already present in the process environment take precedence.
    """
    global _env_loaded
    if _env_loaded:
        return

    # Walk up from this module's directory to find project root
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    dotenv_path = os.path.join(project_root, ".env")

    try:
        from dotenv import load_dotenv

        if os.path.isfile(dotenv_path):
            _ = load_dotenv(dotenv_path, override=False)
            _log.debug("Loaded .env from %s", dotenv_path)
        else:
            _log.warning(".env not found at %s — using process env vars only", dotenv_path)
        # pip 部署时包旁无 .env——从标准用户位置补充（~/.config/memory-skill/.env）
        user_dotenv = os.path.join(os.path.expanduser("~"), ".config", "memory-skill", ".env")
        if os.path.isfile(user_dotenv):
            _ = load_dotenv(user_dotenv, override=False)
            _log.debug("Loaded .env from %s", user_dotenv)
    except ImportError:
        _log.warning("python-dotenv not installed — using process env vars only")

    _env_loaded = True


def _get_env(key: str, default: str) -> str:
    """Read an environment variable, ensuring ``.env`` is loaded first."""
    _load_env()
    return os.environ.get(key, default)


def _sanitise_label(raw: str) -> str | None:
    """Extract a valid label from raw LLM output.  Returns None if none found."""
    raw_lower = raw.strip().lower()

    # Direct match
    if raw_lower in _VALID_LABELS:
        return raw_lower

    # Substring scan
    for label in _VALID_LABELS:
        if label in raw_lower:
            return label

    return None


def _classify_via_api(content: str, model: str) -> str | None:
    """Classify *content* via the DeepSeek API.  Returns label or None on failure.

    Delegates the HTTP call to the shared ``_llm_utils.call_llm`` so there is
    one client/retry/reasoning implementation across the module.

    Security:
      - Content sent to API is truncated to 500 chars.
      - Full API key is NEVER logged — only the masked version.
      - Logged message content is truncated to 60 chars.
    """
    from memory_skill._llm_utils import call_llm

    api_base = _get_env("IMPORTANCE_API_BASE", _DEFAULT_API_BASE)
    api_key = _get_env("IMPORTANCE_API_KEY", _DEFAULT_API_KEY)
    truncated = content[:500]

    try:
        raw = call_llm(
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompt=_CLASSIFICATION_TEMPLATE.format(content=truncated),
            max_tokens=1024,
            temperature=0.0,
            retries=0,
        )
    except Exception as exc:
        _log.warning("LLM gate failed: %s, falling back to rules", exc)
        return None

    label = _sanitise_label(raw or "")

    if label is None:
        _log.warning(
            "LLM gate returned unrecognized label=%r (from msg=%r), falling back to rules",
            (raw or "")[:60],
            truncated[:60],
        )
        return None

    _log.debug(
        "LLM classified msg=%r as %s (raw=%r)",
        truncated[:60],
        label,
        raw.strip()[:60],
    )
    return label


def _classify_via_rules(content: str) -> str:
    """Classify *content* using the rule-based ``ImportanceScorer``.

    Scoring map (based on ``ImportanceScorer.evaluate()`` output):
        score >= 0.7 → "critical"
        score >= 0.3 → "important"
        else         → "trivial"
    """
    from memory_skill.importance import ImportanceScorer

    scorer = ImportanceScorer()
    score, _persist = scorer.evaluate(content)

    if score >= 0.7:
        return "critical"
    elif score >= 0.3:
        return "important"
    else:
        return "trivial"


# ═══════════════════════════════════════════════════════════════════════════════
# LLMImportanceGate
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LLMImportanceGate:
    """LLM-based importance classifier with automatic rules fallback.

    Parameters
    ----------
    threshold:
        Messages with score below this value are considered too trivial
        for long-term storage.  Default from ``IMPORTANCE_THRESHOLD`` env
        var (or 0.3).
    """

    threshold: float = field(default_factory=lambda: float(
        _get_env("IMPORTANCE_THRESHOLD", _DEFAULT_THRESHOLD)
    ))

    def evaluate(self, content: str) -> tuple[float, bool, str]:
        """Evaluate content importance.

        Returns
        -------
        (score, persist, category)
            *score* is in [0.0, 1.0] mapped from the category.
            *persist* is ``True`` when score >= threshold.
            *category* is one of ``"trivial"`` / ``"important"`` / ``"critical"``.
        """
        # ── Fast path: empty content ──────────────────────────────────────
        if not content or not content.strip():
            return (0.0, False, "trivial")

        model = _get_env("IMPORTANCE_MODEL", _DEFAULT_MODEL)

        # ── Try LLM first ─────────────────────────────────────────────────
        label = _classify_via_api(content, model)

        if label is None:
            # ── Fallback to rules ─────────────────────────────────────────
            label = _classify_via_rules(content)
            _log.info(
                "LLM gate fell back to rules for msg=%r → %s",
                content.strip()[:60],
                label,
            )

        score = _CATEGORY_SCORES.get(label, 0.5)
        persist = score >= self.threshold
        return (score, persist, label)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════


def _test() -> None:
    """Run a quick smoke-test through the gate.

    Uses 10 hand-picked test cases covering all three categories.
    Prints ✅/❌ for each case.
    """
    cases: list[tuple[str, str]] = [
        ("你好", "trivial"),
        ("ok", "trivial"),
        ("今天天气不错", "trivial"),
        ("嗯嗯", "trivial"),
        ("我在用 Python 3.13", "important"),
        ("已经修好了那个 bug", "important"),
        ("记住：数据库密码是 abc123", "critical"),
        ("我讨厌这个配色", "critical"),
        ("我用的是 Ubuntu 22.04", "important"),
        ("别忘了明天下午三点开会", "critical"),
    ]

    gate = LLMImportanceGate()
    passed = 0
    total = len(cases)

    print(f"\n{'='*60}")
    print(f"  LLMImportanceGate smoke test  ({total} cases)")
    print(f"  model={_get_env('IMPORTANCE_MODEL', _DEFAULT_MODEL)!r}")
    print(f"  threshold={gate.threshold}")
    print(f"{'='*60}\n")

    for i, (content, expected) in enumerate(cases, 1):
        score, persist, category = gate.evaluate(content)
        ok = category == expected
        if ok:
            passed += 1

        symbol = "✅" if ok else "❌"
        display = content if len(content) <= 40 else content[:37] + "..."
        msg = (
            f"  {symbol}  [{i:2d}]  {display:<42s}  "
            f"got={category:<10s}  expected={expected:<10s}  "
            f"score={score:.2f}  persist={persist}"
        )
        print(msg)

    print(f"\n  Result: {passed}/{total} passed\n")

    if passed == total:
        print("  ✅ All smoke tests passed!\n")
    else:
        print(f"  ❌ {total - passed} test(s) failed — check LLM configuration.\n")


# ── Direct execution ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    _test()
