## Quick Start
Sample code from https://huggingface.co/microsoft/phi-2#sample-code

```bash
git clone https://github.com/yf23/phi-2-test.git
cd phi-2-test
git pull
sudo docker run --gpus all -v /home/yfu/phi-2-test:/home -it --rm nvcr.io/nvidia/pytorch:23.09-py3 python -m pip uninstall -y transformers && python -m pip install git+https://github.com/huggingface/transformers && python phi-2-quickstart.py
```
