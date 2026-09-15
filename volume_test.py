import modal
from pathlib import Path

app = modal.App("volume-test")

volume = modal.Volume.from_name("huggingface-cache")


@app.function(
    volumes={"/cache": volume}
)
def write_test():
    path = Path("/cache/test.txt")
    path.write_text("Persistent Modal storage is working.\n")

    volume.commit()

    print("Wrote:", path)
    print(path.read_text())


@app.local_entrypoint()
def main():
    write_test.remote()