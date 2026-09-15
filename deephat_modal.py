import modal

app = modal.App("deephat-7b")

hf_cache = modal.Volume.from_name(
    "huggingface-cache"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "numpy",
    )
)

MODEL_ID = "DeepHat/DeepHat-V1-7B"
CACHE_DIR = "/root/.cache/huggingface"


@app.function(
    image=image,
    gpu="T4",
    timeout=1200,
    volumes={
        CACHE_DIR: hf_cache,
    },
)
def test_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    print(f"Loading {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )

    messages = [
        {
            "role": "user",
            "content": (
                "In an authorized web application security assessment, "
                "explain three common causes of IDOR vulnerabilities."
            ),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print("\nMODEL RESPONSE:\n")
    print(response)

    hf_cache.commit()


@app.local_entrypoint()
def main():
    test_model.remote()