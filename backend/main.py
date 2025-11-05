"""
Optimizer Backend (Flask)
=========================

A lightweight Flask service that:
  1) Accepts C code via POST /optimize
  2) Calls a local/remote LLM (Ollama-compatible /v1/chat/completions)
  3) Validates the LLM's JSON against a strict schema
  4) Returns a compact response: {optimized_code, suggestions[], metrics{}}

Environment variables:
  OLLAMA_BASE  (default: http://127.0.0.1:11434)
  MODEL_NAME   (default: qwen2.5-coder:7b)
  OPT_DEBUG    (set "1" to print raw LLM text)

Run (dev):
  pip install flask httpx jsonschema
  FLASK_APP=app.py flask run --host 127.0.0.1 --port 8000

Run (prod-ish):
  pip install waitress
  waitress-serve --host=127.0.0.1 --port=8000 app:app

Request example:
  curl -X POST http://127.0.0.1:8000/optimize \
       -H "Content-Type: application/json" \
       -d '{"language":"c","code":"int main(){return 0;}\n"}'
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List

from flask import Flask, jsonify, request
import httpx
from jsonschema import validate, ValidationError

# ---------------------- Configuration ----------------------

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")  # default local
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5-coder:7b")
DEBUG_LOG = os.environ.get("OPT_DEBUG", "0") == "1"

app = Flask(__name__)


# ---------------------- LLM JSON Schema ----------------------

# Expected shape of the LLM response (STRICT).
# - We ask the model to produce ONLY this JSON (no prose).
# - Findings include line/evidence and an optional patch; we only surface a minimal subset.
LLM_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["findings", "model_info"],
    "properties": {
        "model_info": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"}
            }
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "function", "title", "severity", "evidence", "suggestion"],
                "properties": {
                    "id": {"type": "string"},
                    "function": {"type": "string"},
                    "title": {"type": "string"},
                    "severity": {"type": "string", "enum": ["info", "low", "medium", "high"]},
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "lines": {"type": "array", "items": {"type": "integer"}},
                            "reason": {"type": "string"}
                        }
                    },
                    "suggestion": {
                        "type": "object",
                        "required": ["summary"],
                        "properties": {
                            "summary": {"type": "string"},
                            "patch": {"type": "string"},
                            "risk": {"type": "string", "enum": ["low", "medium", "high"]}
                        }
                    }
                }
            }
        }
    }
}


# ---------------------- Helpers ----------------------

def build_prompt(code: str) -> List[Dict[str, str]]:
    """
    Build a strict, JSON-only chat prompt to a coder model.
    We instruct the model:
      - C11 portable, no undefined behavior
      - keep semantics identical
      - return ONLY valid JSON matching our schema
    """
    system = (
        "You are a C11 portability/safety reviewer. "
        "NO undefined behavior. Keep program semantics identical. "
        "Return VALID JSON ONLY matching the schema; do not include prose. "
        "When safe, include a GNU unified diff with headers --- a/<file> and +++ b/<file>."
    )
    user_payload = {
        "constraints": ["no_UB", "portable_c11", "json_only", "bounded<=8KB"],
        "file": "active-buffer.c",
        "code_slice": code,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload)},
    ]


def call_ollama(messages: List[Dict[str, str]]) -> str:
    """
    Call an Ollama-compatible /v1/chat/completions endpoint synchronously.
    Raises a 502-like error (returned as JSON) if the call fails.
    """
    url = f"{OLLAMA_BASE}/v1/chat/completions"
    body = {
        "model": MODEL_NAME,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1024,
        "messages": messages,
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
            r = client.post(url, json=body)
    except httpx.RequestError as e:
        return _fail(502, f"LLM request error: {e}")

    if r.status_code != 200:
        return _fail(502, f"LLM HTTP {r.status_code}: {truncate(r.text, 200)}")

    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if DEBUG_LOG:
        print("RAW LLM TEXT:", text[:4000])
    return text


def validate_llm_json(text: str) -> Dict[str, Any]:
    """
    Ensure model output is valid JSON and matches LLM_JSON_SCHEMA.
    Also clamps overly-large responses (basic guard).
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _fail(502, "Model did not return valid JSON")

    try:
        validate(instance=obj, schema=LLM_JSON_SCHEMA)
    except ValidationError as e:
        return _fail(502, f"JSON schema invalid: {e.message}")

    if len(text.encode("utf-8")) > 16_000:
        return _fail(502, "Model response too large")

    return obj


def extract_minimal(obj: Dict[str, Any], original_code: str) -> Dict[str, Any]:
    """
    Map the validated LLM object to a compact response that the VS Code extension expects.
    - We do NOT apply the patch on the backend.
    - We keep suggestions small and count findings in metrics.
    """
    findings = obj.get("findings") or []
    out_suggestions: List[Dict[str, Any]] = []

    for f in findings[:5]:  # keep it small
        sug = (f.get("suggestion") or {})
        out_suggestions.append({
            "id": f.get("id", "LLM-001"),
            "title": f.get("title", "Optimization"),
            "severity": f.get("severity", "low"),
            "summary": sug.get("summary"),
            "has_patch": bool(sug.get("patch")),
        })

    metrics = {"model_info": obj.get("model_info", {}), "findings_count": len(findings)}
    return {
        "optimized_code": original_code,
        "suggestions": out_suggestions,
        "metrics": metrics,
    }


def _fail(status_code: int, message: str) -> None:
    """
    Helper for uniform JSON errors. Raises a Flask HTTP response immediately.
    """
    response = jsonify({"detail": message})
    response.status_code = status_code
    # Abort by returning a response (used in helper functions)
    # We raise a RuntimeError to short-circuit execution, caught in the route.
    raise RuntimeError(response.get_data(as_text=True))


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


# ---------------------- Flask Routes ----------------------

@app.errorhandler(RuntimeError)
def handle_runtime_error(err):
    """
    Catch our internal _fail() short-circuit and convert to a real JSON response.
    """
    # Attempt to parse the JSON from the message; fallback to plain text
    msg = str(err)
    try:
        payload = json.loads(msg)
        detail = payload.get("detail", "Internal error")
        status = 502 if "LLM" in detail else 400
        return jsonify({"detail": detail}), status
    except Exception:
        return jsonify({"detail": msg}), 400


@app.post("/optimize")
def optimize():
    """
    POST /optimize
    --------------
    Body (JSON):
      {
        "language": "c",          # optional; currently ignored, defaults to "c"
        "code": "/* C code */"    # required
      }

    Returns (JSON):
      {
        "optimized_code": "<string>",
        "suggestions": [
          {"id": "...", "title": "...", "severity": "low|medium|high|info",
           "summary": "...", "has_patch": true|false}
        ],
        "metrics": {"model_info": {...}, "findings_count": <int>}
      }

    Error (JSON):
      {"detail": "<message>"} with appropriate HTTP status.
    """
    ct = request.headers.get("Content-Type", "")
    raw = request.get_data(cache=False, as_text=False)
    print(f"[optimize] CT={ct!r} bytes={len(raw)} preview={raw[:80]!r}")

    # --- parse JSON robustly ---
    try:
        # Try normal JSON first (allows application/json; charset=utf-8, etc.)
        payload = request.get_json(silent=False, force=False)
    except Exception:
        # If the client sent odd headers, fall back to force parse
        try:
            payload = request.get_json(silent=False, force=True)
        except Exception:
            return jsonify({"detail": "Invalid JSON body"}), 400

    code = (payload.get("code") or "").strip()
    _lang = payload.get("language", "c")

    if not code:
        return jsonify({"detail": "Empty code"}), 400

    # 1) Build prompt
    messages = build_prompt(code)

    # 2) Call LLM
    try:
        text = call_ollama(messages)
    except RuntimeError as e:
        # Raised by _fail() inside call_ollama
        return handle_runtime_error(e)

    # 3) Validate JSON from LLM
    try:
        obj = validate_llm_json(text)
    except RuntimeError as e:
        return handle_runtime_error(e)

    # 4) Reduce to minimal response
    resp = extract_minimal(obj, code)
    return jsonify(resp), 200


# ---------------------- Entrypoint ----------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
