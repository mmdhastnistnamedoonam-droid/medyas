"""Thin LLM client with tool-calling for two API styles:
  anthropic -> POST {base}/v1/messages        (Claude)
  openai    -> POST {base}/chat/completions   (OpenAI and every OpenAI-compatible service)

Neutral message format used by the agent:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "input"}]}
  {"role": "tool", "tool_call_id": str, "content": str}
Tools: [{"name", "description", "schema"}]  (schema = JSON schema of the arguments)
"""
import json

import httpx

DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}


class AIError(Exception):
    pass


def configured(cfg: dict) -> bool:
    return bool(cfg.get("ai_api_key", "").strip())


async def _request(cfg: dict, url: str, headers: dict, body: dict) -> dict:
    key = cfg["ai_api_key"].strip()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=20)) as c:
            r = await c.post(url, headers=headers, json=body)
    except httpx.HTTPError as e:
        raise AIError(f"اتصال به سرویس هوش مصنوعی برقرار نشد ({type(e).__name__})") from None
    try:
        j = r.json()
    except ValueError:
        raise AIError(f"پاسخ نامعتبر از سرویس هوش مصنوعی (HTTP {r.status_code})") from None
    if r.status_code >= 400:
        err = j.get("error") if isinstance(j, dict) else None
        msg = (err.get("message") if isinstance(err, dict) else err) or r.text[:200]
        raise AIError(str(msg).replace(key, "***")[:300])
    return j


# ------------------------------------------------------------------ anthropic
def _to_anthropic(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "…"}]})
        else:  # tool result -> user turn; consecutive results are merged into one turn
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


async def _anthropic(cfg, system, messages, tools, max_tokens):
    base = (cfg.get("ai_base_url") or "https://api.anthropic.com").rstrip("/")
    body = {"model": cfg.get("ai_model") or DEFAULT_MODEL["anthropic"], "max_tokens": max_tokens,
            "system": system, "messages": _to_anthropic(messages)}
    if tools:
        body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["schema"]}
                         for t in tools]
    j = await _request(cfg, f"{base}/v1/messages",
                       {"x-api-key": cfg["ai_api_key"].strip(), "anthropic-version": "2023-06-01",
                        "content-type": "application/json"}, body)
    text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    calls = [{"id": b["id"], "name": b["name"], "input": b.get("input") or {}}
             for b in j.get("content", []) if b.get("type") == "tool_use"]
    return {"text": text.strip(), "tool_calls": calls}


# ------------------------------------------------------------------ openai-compatible
def _to_openai(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            d = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                d["tool_calls"] = [{"id": tc["id"], "type": "function",
                                    "function": {"name": tc["name"],
                                                 "arguments": json.dumps(tc["input"], ensure_ascii=False)}}
                                   for tc in m["tool_calls"]]
            out.append(d)
        else:
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out


async def _openai(cfg, system, messages, tools, max_tokens):
    base = (cfg.get("ai_base_url") or "https://api.openai.com/v1").rstrip("/")
    body = {"model": cfg.get("ai_model") or DEFAULT_MODEL["openai"],
            "messages": _to_openai(system, messages)}
    body["max_completion_tokens" if "api.openai.com" in base else "max_tokens"] = max_tokens
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                           "parameters": t["schema"]}} for t in tools]
    j = await _request(cfg, f"{base}/chat/completions",
                       {"Authorization": f"Bearer {cfg['ai_api_key'].strip()}",
                        "Content-Type": "application/json"}, body)
    try:
        msg = j["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise AIError("ساختار پاسخ سرویس هوش مصنوعی ناشناخته است") from None
    calls = []
    for tc in msg.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except ValueError:
            args = {}
        calls.append({"id": tc["id"], "name": tc["function"]["name"], "input": args})
    return {"text": (msg.get("content") or "").strip(), "tool_calls": calls}


async def chat(cfg: dict, system: str, messages: list[dict], tools: list[dict] | None = None,
               max_tokens: int = 1500) -> dict:
    """-> {"text": str, "tool_calls": [{"id","name","input"}]}"""
    if not configured(cfg):
        raise AIError("هوش مصنوعی هنوز فعال نشده است")
    fn = _anthropic if (cfg.get("ai_provider") or "anthropic") == "anthropic" else _openai
    return await fn(cfg, system, messages, tools, max_tokens)
