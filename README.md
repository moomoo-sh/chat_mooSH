# Chat_mooSH
Chat with a local language model and make it execute commands on your computer!

This is HEAVILY inspired by Victor Taelin's ChatSH : https://github.com/VictorTaelin/ChatSH

The main difference is that it all runs locally using llama-cpp-python.

## Installation
1. Install llama-cpp-python:
```bash
# CPU
pip install llama-cpp-python

# or with CUDA support
pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=on"  
```
2. Download models in gguf format then run the python script:
```bash
python chat_mooSH.py --n_ctx 6144 --model_path models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf 
```
