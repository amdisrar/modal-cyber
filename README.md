# Modal Cyber Lab

A low-cost, reproducible lab for experimenting with GPU-hosted cybersecurity-focused LLMs, AI security, and agent security using Modal.

The project is intended to keep local setup lightweight while using serverless GPUs only when required. Model weights are stored in persistent Modal storage so GPU containers can scale down without losing downloaded models.

## Current Status

- Modal workspace configured and tested
- NVIDIA T4 GPU smoke test completed successfully
- Persistent Modal Volume created: `huggingface-cache`
- Volume persistence verified across container shutdowns
- `DeepHat/DeepHat-V1-7B` tested successfully on a Modal T4
- DeepHat used approximately 11.87 GB of GPU memory during the initial test, with some parameters offloaded to CPU
- Next milestone: authenticated OpenAI-compatible HTTPS endpoint for DeepSeek Harness and other clients

## Architecture

```text
Local PC / WSL2
      |
      | Modal CLI / HTTPS
      v
Modal Serverless GPU
      |
      +-- DeepHat / other Hugging Face models
      |
      +-- Persistent Volume: huggingface-cache
      |
      +-- Scale to zero when idle
```

Planned remote-access flow:

```text
Home PC / Office PC / DeepSeek Harness
                 |
                 | HTTPS
                 v
        Modal OpenAI-compatible API
                 |
                 v
        Cybersecurity-focused LLM
                 |
                 v
              GPU
```

## Current Model

### DeepHat

- Model: `DeepHat/DeepHat-V1-7B`
- Purpose: cybersecurity-oriented LLM experimentation
- Initial GPU: NVIDIA T4
- Persistent Hugging Face cache: `huggingface-cache`

The current script is an inference test. It is not yet a persistent API service.

## Repository Files

```text
.
├── gpu_test.py        # Modal GPU/CUDA smoke test
├── volume_test.py     # Persistent Modal Volume test
├── deephat_modal.py   # DeepHat 7B inference test
├── pyproject.toml     # Local project configuration
├── uv.lock            # Reproducible uv dependency lock
└── src/               # Python package source
```

## Local Requirements

Recommended environment:

- Windows 10/11 with WSL2, or Linux
- Ubuntu WSL/Linux
- Git
- `uv`
- Modal account

## Setup

Clone the repository:

```bash
git clone git@github.com:amdisrar/modal-cyber.git
cd modal-cyber
```

Install the local dependencies:

```bash
uv sync
```

Authenticate with Modal:

```bash
uv run modal setup
```

Confirm the connected Modal resources:

```bash
uv run modal volume list
```

The existing workspace should show:

```text
huggingface-cache
```

## GPU Smoke Test

Run:

```bash
uv run modal run gpu_test.py
```

The test verifies:

- CUDA availability
- Assigned GPU model
- Available GPU VRAM

## Persistent Storage Test

Run:

```bash
uv run modal run volume_test.py
```

Then inspect the persistent volume:

```bash
uv run modal volume ls huggingface-cache
```

The test writes a small file to the Modal Volume and confirms that the data survives after the container exits.

## DeepHat Test

Run:

```bash
uv run modal run deephat_modal.py
```

The first run may need to download model weights from Hugging Face. The weights are cached in the persistent Modal Volume so future GPU containers can reuse them.

The GPU function stops when the local entry point finishes, so the GPU is not intended to remain allocated continuously.

## Data Persistence

The project separates local code, cloud model storage, and temporary GPU execution:

```text
GitHub
  -> source code and configuration

Local WSL
  -> working copy and Modal CLI credentials

Modal Volume
  -> Hugging Face model cache and other persistent model data

Modal GPU container
  -> temporary runtime and GPU memory
```

Do not rely on a Modal container filesystem for persistent data. Store anything that must survive container shutdowns in a Modal Volume or another durable storage service.

## Cost Strategy

The goal is to keep the lab as close to free as practical.

- Use serverless GPU functions instead of an always-on GPU VM
- Allow GPU containers to scale to zero when idle
- Keep large Hugging Face downloads in a persistent Modal Volume
- Use T4-class GPUs where possible
- Test quantized models when larger models do not fit efficiently
- Use larger GPUs only for short experiments that require additional VRAM

## Planned Work

### Model Serving

- Convert DeepHat into an authenticated HTTPS endpoint
- Provide an OpenAI-compatible `/v1/chat/completions` interface
- Connect DeepSeek Harness directly to Modal
- Test access from multiple PCs without SSH tunneling
- Configure scale-to-zero behavior

### Model Experiments

- Compare DeepHat FP16 with a quantized version on T4
- Compare T4 performance with L4 when needed
- Test other cybersecurity-focused and less-restricted models
- Measure quality, latency, VRAM usage, and cost

### AI Security Lab

The longer-term goal is to build a reproducible AI security playground covering:

- Prompt injection
- Jailbreak and model-behavior testing
- RAG security and document poisoning
- Memory poisoning
- Agent and tool security
- MCP security
- Multi-agent security
- Data leakage and exfiltration paths
- OWASP GenAI / LLM risks
- OWASP Agentic AI security risks
- Automated red-team testing with tools such as garak and Promptfoo
- Vulnerable and hardened agent implementations

Potential intentionally vulnerable lab projects include OWASP FinBot, Damn Vulnerable LLM Agent, and other open AI-security training environments.

## Security Notes

Never commit secrets or credentials to this repository.

Do not commit:

```text
.modal.toml
.env
.env.*
SSH private keys
Modal tokens
Hugging Face tokens
API keys
```

Keep remotely accessible model endpoints authenticated. An unauthenticated GPU endpoint can allow other users to consume compute credits.

## Project Goal

Build a practical, low-cost AI and agent-security lab that can be reproduced, tested, improved, and eventually shared with the community for learning and defensive security research.
