#!/bin/bash

# System install python venv required for Ubuntu.
#sudo apt install python3.12-venv

python3 -m venv python-env
source python-env/bin/activate

# Requirements used for development andbuilding requirements
pip install ruff
#pip install lark
#pip3 install --upgrade lark
#pip install pycuda
#pip install numpy
#pip install tqdm

# Ollama
#pip install ollama

# Install required packages
#pip install -r requirements.txt
#pip freeze > requirements.txt

ruff check tensorlang/

# Test Runner for all tests
#python3 tests/runner.py
