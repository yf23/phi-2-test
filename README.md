## Quick Start
Sample code from https://huggingface.co/microsoft/phi-2#sample-code

```bash
cd ~
git clone https://github.com/yf23/phi-2-test.git
cd phi-2-test
git pull
sudo docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all -v ~/phi-2-test:/home/phi-2-test --workdir /home/phi-2-test nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash -c "python -m pip install git+https://github.com/huggingface/transformers && python /home/phi-2-qa-with-context.py"
```

## Benchmark
```bash
cd ~
git clone https://github.com/yf23/phi-2-test.git
cd phi-2-test
git pull
sudo docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all -v ~/phi-2-test:/home/phi-2-test --workdir /home/phi-2-test nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash -c "python -m pip install git+https://github.com/huggingface/transformers && python /home/phi-2-benchmark.py"
```
