import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_TOKEN_LENGTH = 1024
OUTPUT_TOKEN_LENGTH = 256
BATCH_SIZE = 64
N_ITERATIONS = 100

# Benchmark time placeholder
time_tokenizing_list = []
time_generation_list = []

# Load the model and tokenizer
torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# User input question
text = "Briefly summarize about the difference between NC and NCC H100 v5 VMs."
prompt = " ".join([text for _ in range(300)])

# Make batch
batch_prompt = [prompt for _ in range(BATCH_SIZE)]

for _ in range(N_ITERATIONS):
    print(f"ITERATION: {_}/{N_ITERATIONS}")

    # Tokenize prompt
    time_start_tokenizing = time.time()
    input_tokens = tokenizer(
        batch_prompt,
        return_tensors="pt",
        return_attention_mask=False,
        max_length=INPUT_TOKEN_LENGTH,
        truncation=True
    )
    time_end_tokenizing = time.time()
    time_tokenizing = time_end_tokenizing - time_start_tokenizing
    time_tokenizing_list.append(time_tokenizing)

    # Generate output
    time_start_generation = time.time()
    outputs = model.generate(
        **input_tokens,
        pad_token_id=tokenizer.pad_token_id,
        min_new_tokens=OUTPUT_TOKEN_LENGTH,
        max_new_tokens=OUTPUT_TOKEN_LENGTH
    )
    time_end_generation = time.time()
    time_generation = time_end_generation - time_start_generation
    time_generation_list.append(time_generation)

# Print benchmark results
print(f"Average time tokenizing: {sum(time_tokenizing_list) / N_ITERATIONS}")
print(f"Average time generating: {sum(time_generation_list) / N_ITERATIONS}")
