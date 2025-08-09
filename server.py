from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import logging
import os
import json
import uuid
import time
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Environment & logging
# -----------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("server")
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# OpenAI Responses client
# -----------------------------------------------------------------------------
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("OpenAI Python SDK is required. Install: pip install -U openai") from e

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-5")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-5-mini")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", BIG_MODEL)
STREAM_DEBUG = os.environ.get("STREAM_DEBUG", "false").lower() in {"1","true","yes"}

ENABLE_WEB_SEARCH = os.environ.get("ENABLE_WEB_SEARCH", "true").lower() in {"1", "true", "yes"}
# Suggest: small | medium | large
WEB_SEARCH_CONTEXT_SIZE = os.environ.get("WEB_SEARCH_CONTEXT_SIZE")

# -----------------------------------------------------------------------------
# Claude-style schemas (request in / response out)
# -----------------------------------------------------------------------------
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]
    ]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool

class MessagesRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        lv = v.lower()
        if "/" in v:
            provider, name = v.split("/", 1)
            # allow openai/<model>; otherwise pass through the name
            return name
        if "sonnet" in lv:
            return BIG_MODEL
        if "haiku" in lv:
            return SMALL_MODEL
        return v

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def supports_sampling(model_name: str) -> bool:
    # Most Responses-first models (gpt-5, o-series) ignore/forbid temperature/top_p/stop
    m = (model_name or "").lower()
    return not (m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"))

def _parse_tool_result_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for it in content:
            if isinstance(it, dict) and it.get("type") == "text":
                parts.append(it.get("text", ""))
            else:
                try:
                    parts.append(json.dumps(it))
                except Exception:
                    parts.append(str(it))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except Exception:
            return str(content)
    try:
        return str(content)
    except Exception:
        return ""

def _extract_final_text(fr) -> str:
    if not fr:
        return ""
    # 1) direct
    txt = getattr(fr, "output_text", None)
    if txt:
        return txt
    # 2) scan structured output items
    try:
        for item in getattr(fr, "output", []) or []:
            itype = getattr(item, "type", None)
            if itype == "output_text":
                return getattr(item, "text", "") or ""
            if itype == "refusal":
                # expose refusal content so UI isn't blank
                return getattr(item, "text", "") or getattr(item, "reason", "") or "Refused."
            if itype == "summary_text":
                return getattr(item, "text", "") or ""
    except Exception:
        pass
    return ""

# -----------------------------------------------------------------------------
# Anthropic -> OpenAI Responses input
# -----------------------------------------------------------------------------
def _anthropic_to_responses_input(req: MessagesRequest) -> List[Dict[str, Any]]:
    """
    Build OpenAI Responses `input` from Claude-style payload.
    user/system => input_text, assistant => output_text, images => input_image.
    tool_result is folded into a user input_text line (simple single-call flow).
    """
    out: List[Dict[str, Any]] = []

    # System
    if req.system:
        if isinstance(req.system, str):
            out.append({"role": "system", "content": req.system})
        else:
            sys_txt = "\n\n".join([blk.text for blk in req.system if getattr(blk, "type", None) == "text"]) or ""
            if sys_txt:
                out.append({"role": "system", "content": sys_txt})

    # Messages
    for m in req.messages:
        c = m.content
        if isinstance(c, str):
            if m.role == "assistant":
                out.append({"role": "assistant", "content": [{"type": "output_text", "text": c}]})
            else:
                out.append({"role": m.role, "content": [{"type": "input_text", "text": c}]})
            continue

        text_parts: List[Dict[str, Any]] = []
        tool_results: List[ContentBlockToolResult] = []
        text_type = "output_text" if m.role == "assistant" else "input_text"

        for blk in c:  # type: ignore
            btype = getattr(blk, "type", None)
            if btype == "text":
                text_parts.append({"type": text_type, "text": blk.text})
            elif btype == "image":
                src = getattr(blk, "source", {}) or {}
                if isinstance(src, dict):
                    if src.get("type") == "base64":
                        text_parts.append({
                            "type": "input_image",
                            "image_base64": src.get("data", ""),
                            "mime_type": src.get("media_type", "image/png"),
                        })
                    elif src.get("type") in ("url", "image_url"):
                        url = src.get("url") or src.get("image_url")
                        if url:
                            text_parts.append({"type": "input_image", "image_url": url})
            elif btype == "tool_result":
                tool_results.append(blk)
            elif btype == "tool_use":
                # Let the model issue tool calls; we don't pre-inject here.
                pass

        if text_parts:
            out.append({"role": m.role, "content": text_parts})

        if m.role == "user" and tool_results:
            for tr in tool_results:
                out.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Tool result for {tr.tool_use_id}:\n{_parse_tool_result_content(tr.content)}",
                        }
                    ],
                })

    return out

# -----------------------------------------------------------------------------
# Tools mapping
# -----------------------------------------------------------------------------
def _map_tools(req: MessagesRequest) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if ENABLE_WEB_SEARCH:
        tools.append({"type": "web_search_preview"})
    if req.tools:
        for t in req.tools:
            tools.append({
                "type": "function",
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            })
    return tools

def _map_tool_choice(req: MessagesRequest) -> Union[str, Dict[str, Any], None]:
    if not req.tool_choice:
        return None
    choice_type = req.tool_choice.get("type")
    if not choice_type:
        return None
    if choice_type in ("auto", "none"):
        return choice_type
    if choice_type == "tool" and req.tool_choice.get("name"):
        return {"type": "function", "name": req.tool_choice["name"]}
    if choice_type == "web_search_preview":
        return {"type": "web_search_preview"}
    return None

# -----------------------------------------------------------------------------
# OpenAI Responses -> Anthropic message (non-streaming)
# -----------------------------------------------------------------------------
def _responses_to_anthropic(resp: Any, model_name: str) -> MessagesResponse:
    text_out = getattr(resp, "output_text", None) or ""

    tool_uses: List[Dict[str, Any]] = []
    try:
        for item in getattr(resp, "output", []) or []:
            itype = getattr(item, "type", None)
            if itype in ("tool_call", "function_call"):
                call_id = getattr(item, "id", None) or f"tool_{uuid.uuid4().hex[:12]}"
                name = getattr(item, "name", "") or getattr(getattr(item, "function", None), "name", "")
                args = getattr(item, "arguments", None) or getattr(getattr(item, "function", None), "arguments", "{}")
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else (args or {})
                except Exception:
                    parsed_args = {"raw": args}
                tool_uses.append({"type": "tool_use", "id": call_id, "name": name, "input": parsed_args})
    except Exception:
        pass

    content: List[Dict[str, Any]] = []
    if text_out:
        content.append({"type": "text", "text": text_out})
    content.extend(tool_uses)

    usage = getattr(resp, "usage", None) or {}
    in_tok = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0)

    return MessagesResponse(
        id=getattr(resp, "id", f"msg_{uuid.uuid4().hex[:24]}"),
        model=model_name,
        role="assistant",
        content=content or [{"type": "text", "text": ""}],
        stop_reason="end_turn",
        usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
    )

# -----------------------------------------------------------------------------
# Streaming bridge (Responses -> Anthropic SSE)
# -----------------------------------------------------------------------------
def responses_stream_to_anthropic_sse(payload: Dict[str, Any], model_name: str):
    """Proxy OpenAI Responses stream as Anthropic-style SSE without broken strings."""

    def _sse(event: str, data: Any) -> str:
        data_str = data if isinstance(data, str) else json.dumps(data)
        return f"event: {event}\n" + f"data: {data_str}\n\n"

    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model_name,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            },
        )

        text_block_open = False
        any_text_emitted = False
        next_index = 0
        tool_id_to_index: Dict[str, int] = {}
        final_usage: Dict[str, int] = {"output_tokens": 0}

        with client.responses.stream(**payload) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                
                if STREAM_DEBUG:
                    logger.info("Responses stream event: %s", etype)

                if etype == "response.output_text.delta":
                    if not text_block_open:
                        yield _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                        text_block_open = True
                        next_index = 1
                    any_text_emitted = True
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": getattr(event, "delta", "")}})
                    continue

                if etype == "response.output_text.done":
                    if text_block_open:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                        text_block_open = False
                    continue

                # --- REFUSAL as visible text ---
                if etype == "response.refusal.delta":
                    if not text_block_open:
                        yield _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                        text_block_open = True
                        next_index = 1
                    any_text_emitted = True
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": getattr(event, "delta", "")}})
                    continue

                if etype == "response.refusal.done":
                    if text_block_open:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                        text_block_open = False
                    continue

                # --- SUMMARY as visible text (optional but useful) ---
                if etype == "response.summary.delta":
                    if not text_block_open:
                        yield _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                        text_block_open = True
                        next_index = 1
                    any_text_emitted = True
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": getattr(event, "delta", "")}})
                    continue

                if etype == "response.summary.done":
                    if text_block_open:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                        text_block_open = False
                    continue

                if etype == "response.tool_call.created":
                    call_id = getattr(event, "id", None) or f"tool_{uuid.uuid4().hex[:12]}"
                    name = getattr(getattr(event, "function", None), "name", None) or getattr(event, "name", "")
                    idx = tool_id_to_index.setdefault(call_id, (next_index or 1))
                    if idx == (next_index or 1):
                        next_index = idx + 1
                    yield _sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}}})
                    continue

                if etype == "response.tool_call.delta":
                    partial = getattr(event, "delta", None)
                    if partial is None:
                        func = getattr(event, "function", None)
                        if func is not None:
                            partial = getattr(func, "arguments", "")
                    call_id = getattr(event, "id", None) or getattr(event, "tool_call_id", None)
                    idx = tool_id_to_index.get(call_id, 1)
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": partial or ""}})
                    continue

                if etype == "response.tool_call.done":
                    call_id = getattr(event, "id", None) or getattr(event, "tool_call_id", None)
                    idx = tool_id_to_index.get(call_id, 1)
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                    continue

                if etype == "response.completed":
                    try:
                        fr = getattr(stream, "final_response", None)
                        if fr and getattr(fr, "usage", None):
                            final_usage = {"output_tokens": int(getattr(fr.usage, "output_tokens", 0) or 0)}

                        # If we never streamed any text, flush whatever final text exists
                        if not any_text_emitted:
                            final_text = _extract_final_text(fr)
                            if final_text:
                                yield _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                                yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": final_text}})
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                    except Exception:
                        pass

                    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": final_usage})
                    yield _sse("message_stop", {"type": "message_stop"})
                    yield "data: [DONE]\n\n"
                    return


        # Safety valve
        if text_block_open:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
        yield _sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Streaming bridge error: %s", e)
        yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "error"}, "usage": {"output_tokens": 0}})
        yield _sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Claude-Code Proxy (Responses API)")

@app.get("/")
def root():
    return {"message": "Anthropic-compatible proxy backed by OpenAI Responses API"}

@app.post("/v1/messages")
def create_message(req: MessagesRequest, raw: Request):
    print(req.stream)
    try:
        model = req.model or DEFAULT_MODEL
        input_msgs = _anthropic_to_responses_input(req)
        tools = _map_tools(req)
        tool_choice = _map_tool_choice(req)

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_msgs,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if ENABLE_WEB_SEARCH and WEB_SEARCH_CONTEXT_SIZE:
            payload["web_search_options"] = {"search_context_size": WEB_SEARCH_CONTEXT_SIZE}

        if supports_sampling(model):
            if req.temperature is not None:
                payload["temperature"] = req.temperature
            if req.top_p is not None:
                payload["top_p"] = req.top_p
            if req.stop_sequences:
                payload["stop"] = req.stop_sequences

        if req.stream:
            return StreamingResponse(
                responses_stream_to_anthropic_sse(payload, model_name=model),
                media_type="text/event-stream",
            )

        start = time.time()
        resp = client.responses.create(**payload)
        logger.info("OpenAI responses.create completed in %.2fs", time.time() - start)
        converted = _responses_to_anthropic(resp, model)
        return JSONResponse(content=converted.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /v1/messages: %s", e)
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# -----------------------------------------------------------------------------
# Token counting (approximate via tiktoken)
# -----------------------------------------------------------------------------
try:
    import tiktoken
except Exception:
    tiktoken = None

@app.post("/v1/messages/count_tokens")
def count_tokens(req: TokenCountRequest):
    try:
        if tiktoken is None:
            return JSONResponse(content={"input_tokens": 0})

        pieces: List[str] = []
        if req.system:
            if isinstance(req.system, str):
                pieces.append(req.system)
            else:
                pieces.append("\n\n".join([b.text for b in req.system if getattr(b, "type", None) == "text"]))
        for m in req.messages:
            if isinstance(m.content, str):
                pieces.append(m.content)
            else:
                for blk in m.content:  # type: ignore
                    if getattr(blk, "type", None) == "text":
                        pieces.append(getattr(blk, "text", ""))
                    elif getattr(blk, "type", None) == "tool_result":
                        pieces.append(_parse_tool_result_content(getattr(blk, "content", "")))

        text = "\n\n".join([p for p in pieces if p])

        try:
            enc = tiktoken.encoding_for_model(req.model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        tokens = len(enc.encode(text))
        return JSONResponse(content={"input_tokens": tokens})
    except Exception as e:
        logger.exception("count_tokens failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {e}")

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8082"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

