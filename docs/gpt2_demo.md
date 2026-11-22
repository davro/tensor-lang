# GPT-2 demo: running real GPT-2 small (124M) in TensorLang

TensorLang runs the real GPT-2 small model from a generated, unrolled `.tl` program and checks it against a float64 NumPy
reference. This page covers setup and what each step does. For editing the weights, see the weight-editing section of
[weight_space_programming.md](weight_space_programming.md). Run every command from the repository root.

## Requirements

* TensorLang installed and working on an NVIDIA GPU (pycuda and nvcc; see the README). Tested on a GTX 1080 Ti (Pascal, sm_61).
* Python packages for the export step: `safetensors`, `huggingface_hub`, `tokenizers` (`tokenizers` is also needed to turn
  `--prompt` into token ids; with `--ids` you can skip it).
* Network access for the first run: the export downloads the model from Hugging Face (about 500 MB).
* Disk: roughly 1 GB in total (the Hugging Face cache plus about 500 MB of float32 `.npy` files under `build/gpt2/`).
  `build/` is gitignored, so none of this is committed.

## Quick start

```bash
./tools/gpt2_demo.sh                           # installs the packages, exports the weights, generates the program and the reference
python3 tensorlang.py build/gpt2_forward.tl    # run it: prints the predicted next-token id and saves the logits
python3 tools/compare_gpt2_logits.py           # TensorLang logits against the float64 reference
```

`tools/gpt2_demo.sh` stops after generating the program; the last two commands run it and check it. The same thing, step by step:

```bash
pip install safetensors huggingface_hub tokenizers

# 1. Download GPT-2 small and write it as .npy files (about 500 MB download)
python3 tools/export_gpt2_weights.py

# 2. Tokenize the prompt, generate the TensorLang program, and compute the float64 reference
python3 tools/make_gpt2_demo.py --prompt "The capital of France is" --ref-generate 8

# 3. Run the generated program, then compare its logits with the reference
python3 tensorlang.py build/gpt2_forward.tl
python3 tools/compare_gpt2_logits.py
```

## What each step writes

| Step | Script | Writes |
|---|---|---|
| 1 | `tools/export_gpt2_weights.py` | `build/gpt2/`: 148 weight files (`wte.npy`, `wpe.npy`, `h0_qkv_w.npy` ... `lnf_b.npy`), `config.json`, `tokenizer.json` |
| 2 | `tools/make_gpt2_demo.py` | `build/gpt2_forward.tl` (the program: 12 layers, one call of the function `gpt2_block` each, with the token ids written into the text) and `build/gpt2_ref_logits.npy` (the float64 reference logits). It also prints the reference's next token and top-5, and with `--ref-generate N` the reference's greedy continuation |
| 3 | `tensorlang.py`, `tools/compare_gpt2_logits.py` | The program's `save(logits, ...)` writes `gpt2_logits.npy` under `cache/` (for `build/gpt2_forward.tl`: `cache/build/gpt2_forward.tl/build/gpt2_logits.npy`). The comparison passes when the argmax is identical and the maximum difference is under 1e-2 |

Weights are read with `load("build/gpt2/<name>.npy")`, one array per file. The program is unrolled because `load()` takes a string
literal. GPT-2's tanh-approximation GELU is a user-defined function built from `mult`, `add` and `tanh` (see `tests/gelu_tanh.tl`).

Other options of `tools/make_gpt2_demo.py`: `--weights DIR` (default `build/gpt2`), `--ids 464,3139` (token ids instead of a prompt),
`--out FILE`, and `--shapes-only`, which emits the GPT-2 small program without reading any weights, for type checking.

## Generating text

```bash
python3 tools/gpt2_generate.py --prompt "The capital of France is" --tokens 20 --check-reference
```

Generation is driven from the host and keeps the sequence length fixed (prompt plus new tokens). The token ids are read from a
`.npy` instead of being written into the program, so the compiled kernels are reused from one token to the next.

## Cost

Measured on a GTX 1080 Ti with v0.5.8.

* The first run for a new sequence length compiles the kernels with nvcc: about 15 to 20 s (about 21 s for the generation program).
* A later run with the same sequence length reuses them: about 3 s.
* Generation: about 2.9 s per token. There is no KV cache, so each token recomputes the whole sequence.

Changing the prompt in `make_gpt2_demo.py` changes the program text (the token ids are in it), so it recompiles. Only the
generation script, which reads the ids from a file, reuses the compiled kernels across steps.

## What was verified

* Prompt "The capital of France is": the largest absolute logit difference is 1.1e-4, on logits averaging 83 in magnitude, with
  identical argmax at every position and identical top-5.
* Twenty greedy tokens are identical to the float64 reference. Greedy GPT-2 small repeats itself ("the capital of the French
  Republic, and the capital of the French Republic ..."); the reference does the same.
* The reference is a reimplementation of GPT-2 in NumPy, checked against an independent per-head loop. It has not been
  compared with Hugging Face `transformers`.

## Limits

* One sequence per run (no batch dimension), no KV cache, generation on the host.
* The sequence length is part of the program: a different length is a different program and a new compile.
