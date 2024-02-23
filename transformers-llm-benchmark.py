import gc
import time
import torch
import argparse
import transformers
from utils import ThroughputStreamer, write_csv_file
from configs import PERF_BENCHMARK_CONFIG_DICT, LLM_PERF_BENCHMARK_OUTPUT_CSV_COLUMNS
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM performance benchmark")
    parser.add_argument(
        "-s",
        "--test-scenario",
        type=str,
        default="phi-2",
        choices=PERF_BENCHMARK_CONFIG_DICT.keys(),
        help="Test scenario to run",
    )
    return parser.parse_args()


def run_model(model_name, batch_prompt, input_token_length, output_token_length):
    # Load the model
    time_start_model_loading = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    time_end_model_loading = time.perf_counter()
    time_model_loading = time_end_model_loading - time_start_model_loading

    # Load the tokenizer
    time_start_tokenizer_loading = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
    streamer = ThroughputStreamer()
    time_start_generation = time.perf_counter()
    outputs = model.generate(
        **input_tokens,
        pad_token_id=tokenizer.pad_token_id,
        min_new_tokens=output_token_length,
        max_new_tokens=output_token_length,
        streamer=streamer,
    )
    time_end_generation = time.perf_counter()
    streamer.set_latencies(time_start_generation, time_end_generation)
    time_first_token_latency = streamer.first_token_latency()
    time_generation = streamer.generation_latency()
    throughput = streamer.throughput()

    time_start_tokenizing_output = time.perf_counter()
    for output in outputs:
        tokenizer.decode(output)
    time_end_tokenizing_output = time.perf_counter()
    time_tokenizing_output = time_end_tokenizing_output - time_start_tokenizing_output

    # Clean up memory
    del model
    del tokenizer
    del input_tokens
    del outputs
    time.sleep(10)
    gc.collect()
    torch.cuda.empty_cache()

    return (
        time_model_loading,
        time_tokenizer_loading,
        time_tokenizing,
        time_tokenizing_output,
        time_first_token_latency,
        time_generation,
        throughput,
    )


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
    time_tokenization_output_list = []
    time_first_token_latency_list = []
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
            time_tokenization_output,
            time_first_token_latency,
            time_generation,
            throughput_generation,
        ) = run_model(model_name, batch_prompt, input_token_length, output_token_length)
        time_model_loading_list.append(time_model_loading)
        time_tokenizer_loading_list.append(time_tokenizer_loading)
        time_tokenization_list.append(time_tokenization)
        time_tokenization_output_list.append(time_tokenization_output)
        time_first_token_latency_list.append(time_first_token_latency)
        time_generation_list.append(time_generation)

        # Calculate end to end time
        time_e2e = time_tokenization + time_first_token_latency + time_generation + time_tokenization_output
        time_e2e_list.append(time_e2e)

        throughput_generation_list.append(throughput_generation)
        throughput_e2e = batch_size * output_token_length / time_e2e
        throughput_e2e_list.append(throughput_e2e)

        # Report
        if verbose_run:
            print(f"\tEnd to end time: {time_e2e}")
            print(f"\t\tModel loading time: {time_model_loading} seconds")
            print(f"\t\tTokenizer loading time: {time_tokenizer_loading} seconds")
            print(f"\t\tTokenization time: {time_tokenization} seconds")
            print(f"\t\tOutput tokenization time: {time_tokenization_output} seconds")
            print(f"\t\tFirst token latency: {time_first_token_latency} seconds")
            print(f"\t\tGeneration time: {time_generation} seconds")
            print(f"\tThroughput (e2e): {throughput_e2e} tokens/second")
            print(f"\tThroughput (generation): {throughput_generation} tokens/second")

    # Calculate average
    time_e2e_avg = sum(time_e2e_list) / n_iter
    time_model_loading_avg = sum(time_model_loading_list) / n_iter
    time_tokenizer_loading_avg = sum(time_tokenizer_loading_list) / n_iter
    time_tokenization_avg = sum(time_tokenization_list) / n_iter
    time_tokenization_output_avg = sum(time_tokenization_output_list) / n_iter
    time_first_token_latency_avg = sum(time_first_token_latency_list) / n_iter
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
        print(f"\t\tOutput tokenization time: {time_tokenization_output_avg} seconds")
        print(f"\t\tFirst token latency: {time_first_token_latency_avg} seconds")
        print(f"\t\tGeneration time: {time_generation_avg} seconds")
        print(f"\tThroughput (e2e): {throughput_e2e_avg} tokens/second")
        print(f"\tThroughput (generation): {throughput_generation_avg} tokens/second")
        print("\n\n")

    return (
        time_model_loading_avg,
        time_tokenizer_loading_avg,
        time_tokenization_avg,
        time_tokenization_output_avg,
        time_first_token_latency_avg,
        time_generation_avg,
        time_e2e_avg,
        throughput_e2e_avg,
        throughput_generation_avg,
    )


def run_all_benchmark(test_scenario):
    # Write header to CSV
    result_csv_filepath = f'{test_scenario.split("/")[-1]}_perf_data_{time.strftime("%Y%m%d%H%M%S")}.csv'
    write_csv_file(
        ",".join(LLM_PERF_BENCHMARK_OUTPUT_CSV_COLUMNS),
        result_csv_filepath,
        append=False,
    )

    # Read config
    model_name = PERF_BENCHMARK_CONFIG_DICT[test_scenario]["model_name"]
    input_token_length_list = PERF_BENCHMARK_CONFIG_DICT[test_scenario]["input_token_length"]
    output_token_length_list = PERF_BENCHMARK_CONFIG_DICT[test_scenario]["output_token_length"]
    batch_size_list = PERF_BENCHMARK_CONFIG_DICT[test_scenario]["batch_size"]
    n_iterations = PERF_BENCHMARK_CONFIG_DICT[test_scenario]["n_iterations"]

    # Warm up
    run_model(model_name, ["Get Ready"], 2, 1)

    # Run benchmarks
    for input_token_length in input_token_length_list:
        for output_token_length in output_token_length_list:
            for batch_size in batch_size_list:
                results = run_benchmark(
                    model_name,
                    input_token_length,
                    output_token_length,
                    batch_size,
                    n_iterations,
                )
                write_csv_file(
                    f"{model_name},{input_token_length},{output_token_length},{batch_size},{','.join(map(str, results))}",
                    result_csv_filepath,
                )


if __name__ == "__main__":
    torch.set_default_device("cuda")
    transformers.logging.set_verbosity_error()
    args = parse_args()
    run_all_benchmark(args.test_scenario)
