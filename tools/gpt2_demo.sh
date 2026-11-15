#!/bin/bash

pip install safetensors huggingface_hub tokenizers
python3 tools/export_gpt2_weights.py                      # ~500 MB download, writes build/gpt2/
python3 tools/make_gpt2_demo.py --prompt "The capital of France is" --ref-generate 8
