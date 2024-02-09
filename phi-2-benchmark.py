import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-2"
INPUT_TOKEN_LENGTH = 1024
OUTPUT_TOKEN_LENGTH = 256
BATCH_SIZE = 64
N_ITERATIONS = 10


def run_model_benchmark(model_name, batch_prompt):
    # Load the model
    time_start_model_loading = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", trust_remote_code=True
    )
    time_end_model_loading = time.time()
    time_model_loading = time_end_model_loading - time_start_model_loading

    # Load the tokenizer
    time_start_tokenizer_loading = time.time()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    time_end_tokenizer_loading = time.time()
    time_tokenizer_loading = time_end_tokenizer_loading - time_start_tokenizer_loading

    # Tokenize prompt
    time_start_tokenizing = time.time()
    input_tokens = tokenizer(
        batch_prompt,
        return_tensors="pt",
        return_attention_mask=False,
        max_length=INPUT_TOKEN_LENGTH,
        truncation=True,
    )
    time_end_tokenizing = time.time()
    time_tokenizing = time_end_tokenizing - time_start_tokenizing

    # Generate output
    time_start_generation = time.time()
    outputs = model.generate(
        **input_tokens,
        pad_token_id=tokenizer.pad_token_id,
        min_new_tokens=OUTPUT_TOKEN_LENGTH,
        max_new_tokens=OUTPUT_TOKEN_LENGTH,
    )
    time_end_generation = time.time()
    time_generation = time_end_generation - time_start_generation

    return time_model_loading, time_tokenizer_loading, time_tokenizing, time_generation


# Benchmark time placeholder
time_model_loading_list = []
time_tokenizer_loading_list = []
time_tokenizing_list = []
time_generation_list = []
time_e2e_list = []
throughput_e2e_list = []
throughput_generation_list = []

# User input question
text = "Briefly summarize about the difference between NC and NCC H100 v5 VMs."
prompt = " ".join([text for _ in range(300)])

# Make batch
batch_prompt = [prompt for _ in range(BATCH_SIZE)]

# Warm up
run_model_benchmark(MODEL_NAME, batch_prompt)

# Start benchmarking
torch.set_default_device("cuda")
for _ in range(N_ITERATIONS):
    print(f"ITERATION: {_+1}/{N_ITERATIONS}")

    # Run benchmark
    time_model_loading, time_tokenizer_loading, time_tokenizing, time_generation = (
        run_model_benchmark(MODEL_NAME, batch_prompt)
    )
    time_model_loading_list.append(time_model_loading)
    time_tokenizer_loading_list.append(time_tokenizer_loading)
    time_tokenizing_list.append(time_tokenizing)
    time_generation_list.append(time_generation)

    # Calculate end to end time
    time_e2e = (
        time_model_loading + time_tokenizer_loading + time_tokenizing + time_generation
    )
    time_e2e_list.append(time_e2e)

    # Calculate throughput
    throughput_generation = BATCH_SIZE * OUTPUT_TOKEN_LENGTH / time_generation
    throughput_generation_list.append(throughput_generation)
    throughput_e2e = BATCH_SIZE * OUTPUT_TOKEN_LENGTH / time_e2e
    throughput_e2e_list.append(throughput_e2e)

    # Report
    print(f"\tEnd to end time: {time_e2e}")
    print(f"\t\tModel loading time: {time_model_loading}")
    print(f"\t\tTokenizer loading time: {time_tokenizer_loading}")
    print(f"\t\tTokenizing time: {time_tokenizing}")
    print(f"\t\tGeneration time: {time_generation}")
    print(f"\tThroughput (e2e): {throughput_e2e}")
    print(f"\tThroughput (generation): {throughput_generation}")

# Print summary benchmark results
print("\n\nSUMMARY BENCHMARK RESULTS")
print(f"\tEnd to end time: {sum(time_e2e_list)/N_ITERATIONS}")
print(f"\t\tModel loading time: {sum(time_model_loading_list)/N_ITERATIONS}")
print(f"\t\tTokenizer loading time: {sum(time_tokenizer_loading_list)/N_ITERATIONS}")
print(f"\t\tTokenizing time: {sum(time_tokenizing_list)/N_ITERATIONS}")
print(f"\t\tGeneration time: {sum(time_generation_list)/N_ITERATIONS}")
print(f"\tThroughput (e2e): {sum(throughput_e2e_list)/N_ITERATIONS}")
print(f"\tThroughput (generation): {sum(throughput_generation_list)/N_ITERATIONS}")
