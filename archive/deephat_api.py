import json
import time
import uuid
from typing import Any

import modal

app = modal.App("deephat-openai-api")

MODEL_ID = "DeepHat/DeepHat-V1-7B"
CACHE_DIR = "/root/.cache/huggingface"

hf_cache = modal.Volume.from_name("huggingface-cache")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "fastapi",
    )
)


@app.function(
    image=image,
    gpu="T4",
    volumes={CACHE_DIR: hf_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=120,
    timeout=1200,
    startup_timeout=1200,
)
@modal.asgi_app(requires_proxy_auth=True)
def create_api():
    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory allocated:",
            round(torch.cuda.memory_allocated() / 1024**3, 2),
            "GB",
        )

    web_app = FastAPI(title="DeepHat OpenAI-compatible API")

    class ChatMessage(BaseModel):
        role: str
        content: Any

    class ChatCompletionRequest(BaseModel):
        model: str = MODEL_ID
        messages: list[ChatMessage]
        temperature: float = 0.7
        top_p: float = 0.95
        max_tokens: int | None = Field(default=None, ge=1, le=2048)
        max_completion_tokens: int | None = Field(default=None, ge=1, le=2048)
        stream: bool = False
        stop: str | list[str] | None = None
        tools: list[dict[str, Any]] | None = None
        tool_choice: Any | None = None

    @web_app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": MODEL_ID,
            "cuda": torch.cuda.is_available(),
        }

    @web_app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    def generate_text(request: ChatCompletionRequest):
        if request.model not in (MODEL_ID, "deephat-7b"):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model: {request.model}",
            )

        messages = [message.model_dump() for message in request.messages]

        template_kwargs: dict[str, Any] = {
            "conversation": messages,
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if request.tools:
            template_kwargs["tools"] = request.tools

        try:
            inputs = tokenizer.apply_chat_template(**template_kwargs).to(model.device)
        except Exception:
            # Some model chat templates do not accept OpenAI tool definitions.
            template_kwargs.pop("tools", None)
            inputs = tokenizer.apply_chat_template(**template_kwargs).to(model.device)

        max_new_tokens = (
            request.max_completion_tokens
            or request.max_tokens
            or 512
        )

        do_sample = request.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": request.top_p,
        }
        if do_sample:
            generation_kwargs["temperature"] = request.temperature

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                **generation_kwargs,
            )

        prompt_tokens = inputs["input_ids"].shape[-1]
        new_tokens = output[0][prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        completion_tokens = int(new_tokens.shape[-1])
        return text, int(prompt_tokens), completion_tokens

    @web_app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        text, prompt_tokens, completion_tokens = generate_text(request)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if request.stream:
            def event_stream():
                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": text,
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first_chunk)}\n\n"

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
            )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return web_app
