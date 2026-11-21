# GPT-2 demo: running real GPT-2 small (124M) in TensorLang

TensorLang runs the real GPT-2 small model from a generated, unrolled `.tl` program and checks it against a float64 NumPy
reference. This page covers setup and what each step does. For editing the weights, see
[weight_space_programming.md](weight_space_programming.md).

## Requirements

* TensorLang installed and working on an NVIDIA GPU (pycuda and nvcc; see the README). Tested on GTX 1080 Ti (Pascal, sm_61).
* Python packages for the export step: `safetensors`, `huggingface_hub`, `tokenizers`.
* Network access for the first run: the export downloads the model from Hugging Face (about 500 MB).
* Disk: roughly 1 GB in total (the Hugging Face cache plus about 500 MB of float32 `.npy` files under `build/gpt2/`).
  `build/` is gitignored, so nothing here is committed.

## Quick start

```bash
./tools/gpt2_demo.sh
```

This installs the three packages, exports the weights, and generates and runs the demo for the prompt
"The capital of France is". The same thing, step by step:

```bash
pip install safetensors huggingface_hub tokenizers

# 1. Download GPT-2 small and write it as .npy files (about 500 MB download)
python3 tools/export_gpt2_weights.py

# 2. Tokenize a prompt, generate the TensorLang program, and compute the float64 reference
python3 tools/make_gpt2_demo.py --prompt "The capital of France is" --ref-generate 8

# 3. Run the generated program and compare its logits with the reference
python3 tensorlang.py build/gpt2_forward.tl
python3 tools/compare_gpt2_logits.py
```

## What each step creates

| Step | Script | Writes |
|---|---|---|
| 1 | `tools/export_gpt2_weights.py` | `build/gpt2/`: 148 weight files (`wte.npy`, `wpe.npy`, `h0_qkv_w.npy` ... `lnf_b.npy`), `config.json`, `tokenizer.json` |
| 2 | `tools/make_gpt2_demo.py` | `build/gpt2_forward.tl` (the program, 12 layers unrolled), `build/gpt2_tokens.npy` (prompt token ids), `build/gpt2_ref_logits.npy` (float64 reference logits) |
| 3 | `tensorlang.py`, `tools/compare_gpt2_logits.py` | the TensorLang logits under `cache/`, and the comparison |

Weights are loaded with `load("build/gpt2/<name>.npy")`, one array per file, so the program text stays small and the compile
cache stays valid when only the token ids change.

## Cost

Measured on a GTX 1080 Ti with v0.5.8; later versions are not slower.

* The first run for a new sequence length compiles the kernels with nvcc: about 15 to 20 s.
* A later run with the same sequence length reuses the compiled kernels: about 3 s.
* `tools/gpt2_generate.py` generates on the host by re-running the program per token, at about 2.9 s per token. There is no KV
  cache, so each token recomputes the whole sequence.

## What was verified

* Logits match a float64 NumPy reference to about 1.1e-4 (the logits are around 83 in magnitude), with identical argmax.
* 20 greedy tokens are identical to the reference.
* The reference is this project's own NumPy implementation of GPT-2. It has not been compared with Hugging Face
  `transformers`. (Delete this line if that comparison has been done.)

## Limits

* One sequence per run (no batch dimension), no KV cache, generation on the host.
* A new sequence length means a new compile.
* The program is generated for a fixed prompt length; it is not a general `generate()`.
