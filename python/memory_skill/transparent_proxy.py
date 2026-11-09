"""
Memory Skill — Transparent Proxy
=================================

OpenAI-compatible HTTP proxy that transparently:
  1. Injects memory context before each request (auto weave)
  2. Auto-ingests both sides after each response

The Agent points its ``OPENAI_API_BASE`` at this proxy's URL and interacts
with the LLM normally — zero code changes, zero tool calls needed.

Start::

    python -m memory_skill.transparent_proxy

Or in code::

    from memory_skill.transparent_proxy import TransparentProxy
    proxy = TransparentProxy(skill, api_base, api_key, port=8888)
    proxy.start()

Environment variables:
  ``MEMORY_SKILL_DB_PATH`` — SQLite db path (default: memory.db)
  ``MEMORY_SKILL_AGENT`` — agent name for namespace (default: transparent)
  ``DEEPSEEK_API_KEY`` — LLM API key
  ``DEEPSEEK_API_BASE`` — LLM API base URL
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from memory_skill import MemorySkill, MemorySkillConfig

logger = logging.getLogger("memory_skill.proxy")

_DEFAULT_PORT = 8888


class TransparentProxy:
    """OpenAI-compatible proxy with transparent memory injection."""

    def __init__(
        self,
        skill: MemorySkill,
        api_base: str,
        api_key: str,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self._skill = skill
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._port = port
        self._server: HTTPServer | None = None
        self._lock = threading.Lock()

    # ── Public ───────────────────────────────────────────────────────────

    def start(self, block: bool = False) -> None:
        """Start the proxy.

        Parameters
        ----------
        block:
            If ``True``, block the calling thread forever (CLI mode).
            If ``False`` (default), serve on a daemon thread.
        """
        self._server = HTTPServer(
            ("127.0.0.1", self._port),
            self._build_handler(),
        )
        print(f"\n  Transparent Memory Proxy → http://127.0.0.1:{self._port}")
        print(f"  Upstream LLM → {self._api_base}\n")

        if block:
            self._server.serve_forever()
        else:
            t = threading.Thread(target=self._server.serve_forever, daemon=True)
            t.start()
            # Give the server a moment to bind
            import time
            time.sleep(0.1)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    # ── Handler factory ──────────────────────────────────────────────────

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                path = self.path.rstrip("/")
                if path in ("/v1/chat/completions", "/chat/completions"):
                    proxy._chat_completions(self)
                else:
                    proxy._passthrough(self)

            def do_GET(self) -> None:
                proxy._passthrough(self)

            def log_message(self, fmt: str, *args: Any) -> None:
                pass  # suppress HTTP logs

        return _Handler

    # ── Chat completions ─────────────────────────────────────────────────

    def _chat_completions(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(length))
        messages: list[dict] = body.get("messages", [])

        # ── 1. Auto-inject memory context ────────────────────────────
        with self._lock:
            enriched = self._skill.auto_context(messages)

        last_msg = messages[-1]["content"] if messages else ""
        expanded = self._skill.expand(last_msg)
        if expanded and enriched:
            enriched[-1]["content"] = expanded + "\n\n" + enriched[-1]["content"]

        body["messages"] = enriched
        body["stream"] = False  # force non-streaming

        # ── 2. Forward to LLM ───────────────────────────────────────
        url = f"{self._api_base}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
        except Exception as exc:
            self._error(handler, 502, str(exc))
            return

        # ── 3. Auto-ingest both sides ───────────────────────────────
        last_user = next(
            (
                m["content"]
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        assistant = ""
        for choice in result.get("choices", []):
            assistant = choice.get("message", {}).get("content", "") or ""
            break

        if last_user and assistant:
            with self._lock:
                self._skill.auto_ingest(last_user, assistant)

        # ── 4. Return ───────────────────────────────────────────────
        self._json(handler, 200, result)

    # ── Passthrough ─────────────────────────────────────────────────────

    def _passthrough(self, handler: BaseHTTPRequestHandler) -> None:
        url = f"{self._api_base}{handler.path}"
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}

        body: bytes | None = None
        if handler.command == "POST":
            length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(length)
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=handler.command,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                handler.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() != "transfer-encoding":
                        handler.send_header(k, v)
                handler.end_headers()
                handler.wfile.write(resp.read())
        except Exception as exc:
            self._error(handler, 502, str(exc))

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _json(handler: BaseHTTPRequestHandler, code: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _error(handler: BaseHTTPRequestHandler, code: int, msg: str) -> None:
        TransparentProxy._json(handler, code, {"error": msg})


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _serve(port: int = _DEFAULT_PORT) -> None:
    """CLI entry point — start the transparent proxy."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = MemorySkillConfig(
        db_path=os.getenv("MEMORY_SKILL_DB_PATH", "memory.db"),
        agent_name=os.getenv("MEMORY_SKILL_AGENT", "transparent"),
    )
    skill = MemorySkill(config)

    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("DEEPSEEK_API_KEY environment variable is required.")
        return

    proxy = TransparentProxy(skill, api_base, api_key, port)
    proxy.start(block=True)


if __name__ == "__main__":
    _serve()
