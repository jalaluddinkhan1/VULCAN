#!/usr/bin/env python3
"""
VULCAN OpenAI-Compatible API Server

Exposes VULCAN as a drop-in replacement for the OpenAI Chat Completions API.
Tools like LangChain, LlamaIndex, and OpenAI clients work without modification.

Endpoints:
    GET  /v1/models                    — list available model
    POST /v1/chat/completions          — chat completions (streaming + non-streaming)
    POST /v1/completions               — text completions

Usage:
    pip install fastapi uvicorn
    python server.py --model model_q4.vulcan --port 8000

Then point any OpenAI client at http://localhost:8000:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="vulcan")
    resp = client.chat.completions.create(
        model="vulcan-7b",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

import argparse
import json
import time
import uuid
import sys
from typing import List, Optional, Iterator

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("[ERROR] Required packages missing.")
    print("        pip install fastapi uvicorn")
    sys.exit(1)

try:
    import vulcan as _vulcan
except ImportError:
    print("[ERROR] VULCAN Python module not found.")
    print("        Build the project first: cmake --build build")
    sys.exit(1)


# ─── Globals ────────────────────────────────────────────────────────────────

_engine: Optional[_vulcan.Engine] = None
_model_name: str = "vulcan-7b"

# Minimal Llama-2 tokenizer using sentencepiece (if available), else passthrough
_tokenizer = None


def _load_tokenizer(model_path: str):
    """Try to load a sentencepiece tokenizer from the model directory."""
    global _tokenizer
    import os
    sp_path = os.path.join(os.path.dirname(model_path), "tokenizer.model")
    if os.path.exists(sp_path):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(sp_path)
            _tokenizer = sp
            print(f"[VULCAN] Loaded tokenizer: {sp_path}")
        except ImportError:
            print("[VULCAN] sentencepiece not installed — using character fallback")
    else:
        print("[VULCAN] No tokenizer.model found — using character fallback")


def _encode(text: str) -> List[int]:
    """Encode text to token IDs."""
    if _tokenizer is not None:
        return [1] + _tokenizer.Encode(text)  # BOS + tokens
    # Fallback: UTF-8 byte encoding (not real, just for testing)
    return [1] + [b for b in text.encode("utf-8")[:512]]


def _decode(token_ids: List[int]) -> str:
    """Decode token IDs to text, skipping special tokens."""
    # Remove BOS (1) and EOS (2)
    ids = [t for t in token_ids if t not in (0, 1, 2)]
    if _tokenizer is not None:
        return _tokenizer.Decode(ids)
    # Fallback: treat IDs as UTF-8 bytes
    try:
        return bytes([t for t in ids if 0 <= t <= 255]).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _format_chat_prompt(messages: list) -> str:
    """Format chat messages into Llama-2 chat format."""
    # Llama-2 chat format:
    # <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>
    system = ""
    prompt = ""
    first_user = True

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system = content
        elif role == "user":
            if first_user and system:
                prompt += f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{content} [/INST]"
                first_user = False
            else:
                prompt += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content} </s>"

    return prompt


# ─── Pydantic Request/Response Models ────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "vulcan-7b"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = "vulcan-7b"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    stream: bool = False


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="VULCAN Inference API",
    description="OpenAI-compatible REST API for the VULCAN LLM engine",
    version="1.0.0"
)


@app.get("/v1/models")
def list_models():
    """List available models — mirrors OpenAI /v1/models."""
    return {
        "object": "list",
        "data": [{
            "id": _model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "vulcan",
            "permission": [],
            "root": _model_name,
            "parent": None,
        }]
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _engine is not None and _engine.is_ready()}


def _run_generation(prompt_ids: List[int], req_max_tokens: int,
                    temperature: float, top_p: float,
                    top_k: int) -> List[int]:
    """Run engine.generate() with the given parameters."""
    if _engine is None or not _engine.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    gen_cfg = _vulcan.GenerationConfig()
    gen_cfg.max_tokens = req_max_tokens
    gen_cfg.temperature = max(temperature, 1e-6)  # avoid zero temperature
    gen_cfg.top_p = top_p
    gen_cfg.top_k = top_k
    gen_cfg.greedy = (temperature == 0.0)

    _engine.reset_cache()
    all_ids = _engine.generate(prompt_ids, gen_cfg)
    # Return only the newly generated tokens (after the prompt)
    return all_ids[len(prompt_ids):]


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_text = _format_chat_prompt(messages)
    prompt_ids = _encode(prompt_text)

    completion_id = "chatcmpl-" + uuid.uuid4().hex[:8]
    created = int(time.time())

    if req.stream:
        def stream_tokens() -> Iterator[str]:
            # For streaming: generate all tokens, yield one at a time
            try:
                generated_ids = _run_generation(
                    prompt_ids, req.max_tokens,
                    req.temperature, req.top_p, req.top_k
                )
                for token_id in generated_ids:
                    token_text = _decode([token_id])
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": _model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": _model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(error)}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    # Non-streaming
    generated_ids = _run_generation(
        prompt_ids, req.max_tokens, req.temperature, req.top_p, req.top_k
    )
    completion_text = _decode(generated_ids)
    prompt_tokens = len(prompt_ids)
    completion_tokens = len(generated_ids)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": _model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": completion_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    """OpenAI-compatible text completions endpoint."""
    prompt_ids = _encode(req.prompt)

    generated_ids = _run_generation(
        prompt_ids, req.max_tokens, req.temperature, req.top_p, req.top_k
    )
    completion_text = _decode(generated_ids)
    completion_id = "cmpl-" + uuid.uuid4().hex[:8]

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [{
            "text": completion_text,
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(prompt_ids) + len(generated_ids)
        }
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    global _engine, _model_name

    parser = argparse.ArgumentParser(
        description="VULCAN OpenAI-Compatible API Server"
    )
    parser.add_argument("--model", required=True,
                        help="Path to .vulcan model file (e.g., model_q4.vulcan)")
    parser.add_argument("--model-name", default="vulcan-7b",
                        help="Model ID returned by /v1/models (default: vulcan-7b)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--rope-theta", type=float, default=10000.0,
                        help="RoPE base — Llama-2: 10000, Llama-3: 500000")
    args = parser.parse_args()

    _model_name = args.model_name

    # Build model config
    model_cfg = _vulcan.ModelConfig()
    model_cfg.hidden_dim   = args.hidden_dim
    model_cfg.num_heads    = args.num_heads
    model_cfg.num_kv_heads = args.num_kv_heads
    model_cfg.num_layers   = args.num_layers
    model_cfg.vocab_size   = args.vocab_size
    model_cfg.max_seq_len  = args.max_seq_len
    model_cfg.rope_theta   = args.rope_theta

    # Load engine
    print(f"[VULCAN] Loading model: {args.model}")
    _engine = _vulcan.Engine()
    if not _engine.load_model(args.model, model_cfg):
        print("[ERROR] Failed to load model.")
        sys.exit(1)

    cache_mb = _engine.cache_memory_usage() / (1024 * 1024)
    print(f"[VULCAN] Model ready. KV cache: {cache_mb:.0f} MB")

    _load_tokenizer(args.model)

    print(f"[VULCAN] Server starting at http://{args.host}:{args.port}")
    print(f"[VULCAN] OpenAI endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
