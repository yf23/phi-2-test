import gc
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "microsoft/phi-2"
INPUT_TOKEN_LENGTH_LIST = [32, 256, 1024, 2048]
OUTPUT_TOKEN_LENGTH_LIST = [1, 32, 64, 128]
BATCH_SIZE_LIST = [1, 8, 32]
N_ITERATIONS = 5


def write_csv_file(line, filepath, append=True):
    mode = "a" if append else "w"
    with open(filepath, mode) as f:
        f.write(line + "\n")


def run_model(model_name, batch_prompt, input_token_length, output_token_length):
    # Load the model
    time_start_model_loading = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", trust_remote_code=True
    )
    time_end_model_loading = time.perf_counter()
    time_model_loading = time_end_model_loading - time_start_model_loading

    # Load the tokenizer
    time_start_tokenizer_loading = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    time_end_tokenizer_loading = time.perf_counter()
    time_tokenizer_loading = time_end_tokenizer_loading - time_start_tokenizer_loading

    # Tokenize prompt
    time_start_tokenizing = time.perf_counter()
    input_tokens = tokenizer(
        batch_prompt,
        return_tensors="pt",
        return_attention_mask=False,
        max_length=input_token_length,
        truncation=True,
    )
    time_end_tokenizing = time.perf_counter()
    time_tokenizing = time_end_tokenizing - time_start_tokenizing

    # Generate output
    time_start_generation = time.perf_counter()
    outputs = model.generate(
        **input_tokens,
        pad_token_id=tokenizer.pad_token_id,
        min_new_tokens=output_token_length,
        max_new_tokens=output_token_length,
    )
    time_end_generation = time.perf_counter()
    time_generation = time_end_generation - time_start_generation

    # Clean up memory
    del model
    del tokenizer
    del input_tokens
    del outputs
    time.sleep(10)
    gc.collect()
    torch.cuda.empty_cache()

    return time_model_loading, time_tokenizer_loading, time_tokenizing, time_generation


def run_benchmark(
    model_name,
    input_token_length,
    output_token_length,
    batch_size,
    n_iter,
    input_text="",
    verbose_run=False,
    verbose_summary=True,
):
    # Print parameters in one line
    if verbose_summary:
        print(
            f"{model_name}: input_len={input_token_length}, output_len={output_token_length}, batch_size={batch_size}"
        )

    # Benchmark time placeholder
    time_model_loading_list = []
    time_tokenizer_loading_list = []
    time_tokenization_list = []
    time_generation_list = []
    time_e2e_list = []
    throughput_e2e_list = []
    throughput_generation_list = []

    # Input text
    text = "Briefly summarize about the difference between NC and NCC H100 v5 VMs."
    if input_text:
        text = input_text
    prompt = " ".join([text for _ in range(500)])

    # Make batch
    batch_prompt = [prompt for _ in range(batch_size)]

    # Start benchmarking
    for _ in range(n_iter):
        if verbose_run:
            print(f"ITERATION: {_+1}/{n_iter}")

        # Run benchmark
        (
            time_model_loading,
            time_tokenizer_loading,
            time_tokenization,
            time_generation,
        ) = run_model(model_name, batch_prompt, input_token_length, output_token_length)
        time_model_loading_list.append(time_model_loading)
        time_tokenizer_loading_list.append(time_tokenizer_loading)
        time_tokenization_list.append(time_tokenization)
        time_generation_list.append(time_generation)

        # Calculate end to end time
        time_e2e = (
            time_model_loading
            + time_tokenizer_loading
            + time_tokenization
            + time_generation
        )
        time_e2e_list.append(time_e2e)

        # Calculate throughput
        throughput_generation = batch_size * output_token_length / time_generation
        throughput_generation_list.append(throughput_generation)
        throughput_e2e = batch_size * output_token_length / time_e2e
        throughput_e2e_list.append(throughput_e2e)

        # Report
        if verbose_run:
            print(f"\tEnd to end time: {time_e2e}")
            print(f"\t\tModel loading time: {time_model_loading} seconds")
            print(f"\t\tTokenizer loading time: {time_tokenizer_loading} seconds")
            print(f"\t\tTokenization time: {time_tokenization} seconds")
            print(f"\t\tGeneration time: {time_generation} seconds")
            print(f"\tThroughput (e2e): {throughput_e2e} tokens/second")
            print(f"\tThroughput (generation): {throughput_generation} tokens/second")

    # Calculate average
    time_e2e_avg = sum(time_e2e_list) / n_iter
    time_model_loading_avg = sum(time_model_loading_list) / n_iter
    time_tokenizer_loading_avg = sum(time_tokenizer_loading_list) / n_iter
    time_tokenization_avg = sum(time_tokenization_list) / n_iter
    time_generation_avg = sum(time_generation_list) / n_iter
    throughput_e2e_avg = sum(throughput_e2e_list) / n_iter
    throughput_generation_avg = sum(throughput_generation_list) / n_iter

    # Print summary benchmark results
    if verbose_summary:
        if verbose_run:
            print("\n\nSUMMARY BENCHMARK RESULTS")
        print(f"\tEnd to end time: {time_e2e_avg} seconds")
        print(f"\t\tModel loading time: {time_model_loading_avg} seconds")
        print(f"\t\tTokenizer loading time: {time_tokenizer_loading_avg} seconds")
        print(f"\t\tTokenization time: {time_tokenization_avg} seconds")
        print(f"\t\tGeneration time: {time_generation_avg} seconds")
        print(f"\tThroughput (e2e): {throughput_e2e_avg} tokens/second")
        print(f"\tThroughput (generation): {throughput_generation_avg} tokens/second")
        print("\n\n")

    return (
        time_model_loading_avg,
        time_tokenizer_loading_avg,
        time_tokenization_avg,
        time_generation_avg,
        time_e2e_avg,
        throughput_e2e_avg,
        throughput_generation_avg,
    )


if __name__ == "__main__":
    torch.set_default_device("cuda")
    transformers.logging.set_verbosity_error()

    # Write header to CSV
    columns = [
        "model_name",
        "input_token_length",
        "output_token_length",
        "batch_size",
        "model_loading_latency",
        "tokenizer_loading_latency",
        "tokenization_latency",
        "generation_latency",
        "total_latency",
        "throughput_e2e",
        "throughput_generation",
    ]
    result_csv_filepath = (
        f'{MODEL_NAME.split("/")[-1]}_perf_data_{time.strftime("%Y%m%d%H%M%S")}.csv'
    )
    write_csv_file(
        ",".join(columns),
        result_csv_filepath,
        append=False,
    )

    # Warm up
    run_model(MODEL_NAME, ["Get Ready"], 2, 1)

    # Run benchmarks
    for input_token_length in INPUT_TOKEN_LENGTH_LIST:
        for output_token_length in OUTPUT_TOKEN_LENGTH_LIST:
            for batch_size in BATCH_SIZE_LIST:
                results = run_benchmark(
                    MODEL_NAME,
                    input_token_length,
                    output_token_length,
                    batch_size,
                    N_ITERATIONS,
                )
                write_csv_file(
                    f"{MODEL_NAME},{input_token_length},{output_token_length},{batch_size},{','.join(map(str, results))}",
                    result_csv_filepath,
                )
