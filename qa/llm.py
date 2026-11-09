"""Optional vision-LLM second opinion on continuity flags.

The heuristics are cheap but noisy; the LLM only sees the 2 frames behind each
outfit/background flag, so cost stays at a few calls per episode.

Swap the model with env vars, no code change:
    QA_LLM_PROVIDER = gemini | openai_compatible
    QA_LLM_MODEL    = e.g. gemini-2.0-flash | gpt-4o-mini | llama-3.2-11b-vision-preview
    GEMINI_API_KEY  or  OPENAI_API_KEY (+ QA_LLM_BASE_URL for Groq/OpenRouter/Ollama/your gateway)
Uses only the standard library (no SDK install).
"""
import base64
import json
import os
import re
import urllib.request

PROMPT = """You are a continuity checker for a short drama series.
Image 1 and image 2 are two shots that should belong to the SAME scene.
Question: {question}
Ignore camera angle, framing, zoom and lighting differences; those are normal.
Reply with JSON only: {{"consistent": true|false, "reason": "<one short sentence>"}}"""

QUESTIONS = {
    "outfit_change": "Is the person wearing the same clothes in both images?",
    "background_change": "Are both images in the same location / set?",
}


def _b64(jpg_bytes):
    return base64.b64encode(jpg_bytes).decode()


def _post(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def ask(img1, img2, question, provider=None, model=None):
    provider = provider or os.environ.get("QA_LLM_PROVIDER")
    model = model or os.environ.get("QA_LLM_MODEL")
    text = PROMPT.format(question=question)
    if provider == "gemini":
        model = model or "gemini-2.0-flash"
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
               f"?key={os.environ['GEMINI_API_KEY']}")
        payload = {"contents": [{"parts": [
            {"text": text},
            {"inline_data": {"mime_type": "image/jpeg", "data": _b64(img1)}},
            {"inline_data": {"mime_type": "image/jpeg", "data": _b64(img2)}}]}]}
        out = _post(url, payload, {"Content-Type": "application/json"})
        reply = out["candidates"][0]["content"]["parts"][0]["text"]
    elif provider == "openai_compatible":
        base = os.environ.get("QA_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        payload = {"model": model or "gpt-4o-mini", "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(img1)}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(img2)}}]}]}
        out = _post(base + "/chat/completions", payload, {
            "Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "")})
        reply = out["choices"][0]["message"]["content"]
    else:
        return None
    m = re.search(r"\{.*\}", reply, re.S)
    return json.loads(m.group(0)) if m else None


def verify(issues, grab_frame, cfg):
    """Annotate continuity issues with the LLM verdict; downgrade the ones it rejects."""
    provider = cfg.get("llm_provider") or os.environ.get("QA_LLM_PROVIDER")
    if not provider:
        return issues
    for it in issues:
        if it["check"] not in QUESTIONS or "ref_time" not in it:
            continue
        try:
            v = ask(grab_frame(it["ref_time"]), grab_frame(it["frame_time"]), QUESTIONS[it["check"]],
                    provider, cfg.get("llm_model"))
        except Exception as e:  # noqa: BLE001  network/quota errors must not kill the run
            it["llm"] = {"error": str(e)[:200]}
            continue
        if v is None:
            continue
        it["llm"] = v
        if v.get("consistent") is True:
            it["severity"] = "info"
            it["detail"] += f" [vision model disagrees: {v.get('reason', '')}]"
        else:
            it["confidence"] = "high"
    return issues
