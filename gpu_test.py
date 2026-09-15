import modal

app = modal.App("gpu-test")

image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy",
)

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
)
def check_gpu():
    import torch

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )

    return "GPU test completed"


@app.local_entrypoint()
def main():
    result = check_gpu.remote()
    print(result)