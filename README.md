## Quick Start
Sample code from https://huggingface.co/microsoft/phi-2#sample-code

```bash
cd ~
git clone https://github.com/yf23/phi-2-test.git
cd phi-2-test
git pull
sudo docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all -v ~/phi-2-test:/home/phi-2-test --workdir /home/phi-2-test nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash -c "python -m pip install -r requirements.txt && python /home/phi-2-test/phi-2-qa-with-context.py"
```

## Benchmark LLM
```bash
cd ~
git clone https://github.com/yf23/phi-2-test.git
cd phi-2-test
git pull
sudo docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all -v ~/phi-2-test:/home/phi-2-test --workdir /home/phi-2-test nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash -c "python -m pip install -r requirements.txt && python /home/phi-2-test/transformers-llm-benchmark.py -s phi-2"
```

## Benchmark Whisper
```bash
cd ~
git clone https://github.com/yf23/phi-2-test.git
cd phi-2-test
git pull
sudo docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all -v ~/phi-2-test:/home/phi-2-test --workdir /home/phi-2-test nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash -c "python -m pip -r requirements.txt && python /home/phi-2-test/whisper-benchmark.py -s whisper-large-v3-test"
```
