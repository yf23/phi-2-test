## Quick Start
Sample code from https://huggingface.co/microsoft/phi-2#sample-code

```bash
cd ~
git clone https://github.com/yf23/phi-2-test.git
cd phi-2-test
git pull
sudo docker run --gpus all -v ~/phi-2-test:/home -it --rm nvcr.io/nvidia/pytorch:23.09-py3 "python -m pip install git+https://github.com/huggingface/transformers && python3 /home/phi-2-quickstart.py"
```
