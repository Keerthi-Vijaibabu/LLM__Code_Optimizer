from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx, json, os
from jsonschema import validate, ValidationError

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")  # default local
MODEL_NAME  = os.environ.get("MODEL_NAME",  "qwen2.5-coder:7b")
DEBUG_LOG   = os.environ.get("OPT_DEBUG", "0") == "1"

app = FastAPI(title="Optimizer Backend", version="0.1.0")

# --------- Request/Response models ----------
class OptimizeReq(BaseModel):
    language: str = "c"
    code: str

class Suggestion(BaseModel):
    id: str
    title: str
    severity: str
    summary: str | None = None
    has_patch: bool

class OptimizeResp(BaseModel):
    optimized_code: str
    suggestions: list[Suggestion]
    metrics: dict

# --------- LLM schema we expect ----------
LLM_JSON_SCHEMA = {
    "type": "object",
    "required": ["findings", "model_info"],
    "properties": {
        "model_info": {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}, "version": {"type": "string"}}
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
                    "severity": {"type": "string", "enum": ["info","low","medium","high"]},
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
                            "risk": {"type": "string", "enum": ["low","medium","high"]}
                        }
                    }
                }
            }
        }
    }
}

# --------- Helpers ----------
def build_prompt(code: str) -> list[dict]:
    system = (
        "You are a C11 portability/safety reviewer. "
        "NO undefined behavior. Keep program semantics identical. "
        "Return VALID JSON ONLY matching the schema; do not include prose. "
        "When safe, include a GNU unified diff with headers --- a/<file> and +++ b/<file>."
    )
    user_payload = {
        "constraints": ["no_UB","portable_c11","json_only","bounded<=8KB"],
        "file": "active-buffer.c",
        "code_slice": code
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload)}
    ]

async def call_ollama(messages: list[dict]) -> str:
    url = f"{OLLAMA_BASE}/v1/chat/completions"
    body = {
        "model": MODEL_NAME,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1024,
        "messages": messages
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(url, json=body)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LLM HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if DEBUG_LOG:
        print("RAW LLM TEXT:", text[:4000])
    return text

def validate_llm_json(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Model did not return valid JSON")
    try:
        validate(instance=obj, schema=LLM_JSON_SCHEMA)
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=f"JSON schema invalid: {e.message}")
    # Basic response size clamp
    if len(text.encode("utf-8")) > 16_000:
        raise HTTPException(status_code=502, detail="Model response too large")
    return obj

def extract_minimal(obj: dict, original_code: str) -> OptimizeResp:
    findings = obj.get("findings") or []
    out_suggestions: list[Suggestion] = []
    # We don't apply the patch on the backend (VS Code side will do). Return the original text.
    for f in findings[:5]:  # keep it small
        sug = f.get("suggestion") or {}
        out_suggestions.append(Suggestion(
            id=f.get("id","LLM-001"),
            title=f.get("title","Optimization"),
            severity=f.get("severity","low"),
            summary=sug.get("summary"),
            has_patch=bool(sug.get("patch"))
        ))
    metrics = {"model_info": obj.get("model_info", {}), "findings_count": len(findings)}
    return OptimizeResp(optimized_code=original_code, suggestions=out_suggestions, metrics=metrics)

# --------- Endpoint ----------
@app.post("/optimize", response_model=OptimizeResp)
async def optimize(req: OptimizeReq):
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Empty code")

    messages = build_prompt(req.code)
    text = await call_ollama(messages)
    obj = validate_llm_json(text)
    resp = extract_minimal(obj, req.code)
    return resp
