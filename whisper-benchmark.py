import gc
import time
import torch
import argparse
import transformers
import numpy as np
from datasets import load_dataset
from utils import ThroughputStreamer, write_csv_file
from configs import (
    WHISPER_PERF_BENCHMARK_CONFIG_DICT,
    WHISPER_PERF_BENCHMARK_OUTPUT_CSV_COLUMNS,
)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run Whisper performance benchmark")
    parser.add_argument(
        "-s",
        "--test-scenario",
        type=str,
        default="whisper-large-v3-test",
        choices=WHISPER_PERF_BENCHMARK_CONFIG_DICT.keys(),
        help="Test scenario to run",
    )
    return parser.parse_args()


def run_model(
    model_name,
    batch_waveform,
    sampling_rate,
    output_token_length,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the model
    time_start_model_loading = time.perf_counter()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype="auto"
    ).to(device)
    time_end_model_loading = time.perf_counter()
    time_model_loading = time_end_model_loading - time_start_model_loading

    # Load the processor
    time_start_processor_loading = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_name)
    time_end_processor_loading = time.perf_counter()
    time_processor_loading = time_end_processor_loading - time_start_processor_loading

    # Process prompt
    time_start_processing = time.perf_counter()
    input_features = processor(
        batch_waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).to(device)
    time_end_processing = time.perf_counter()
    time_processing = time_end_processing - time_start_processing

    # Generate output
    streamer = ThroughputStreamer()
    time_start_generation = time.perf_counter()
    outputs = model.generate(
        **input_features,
        min_new_tokens=output_token_length,
        max_new_tokens=output_token_length,
        streamer=streamer,
    )
    time_end_generation = time.perf_counter()
    streamer.set_latencies(time_start_generation, time_end_generation)
    time_first_token_latency = streamer.first_token_latency()
    time_generation = streamer.generation_latency()
    throughput = streamer.throughput()

    # Decode output
    time_start_output_decoding = time.perf_counter()
    processor.batch_decode(outputs)
    time_end_output_decoding = time.perf_counter()
    time_output_decoding = time_end_output_decoding - time_start_output_decoding

    # Clean up memory
    del model
    del processor
    del input_features
    del outputs
    time.sleep(10)
    gc.collect()
    torch.cuda.empty_cache()

    return (
        time_model_loading,
        time_processor_loading,
        time_processing,
        time_output_decoding,
        time_first_token_latency,
        time_generation,
        throughput,
    )


def run_benchmark(
    model_name,
    output_token_length,
    batch_size,
    n_iter,
    verbose_run=False,
    verbose_summary=True,
):
    # Print parameters in one line
    if verbose_summary:
        print(
            f"{model_name}: output_len={output_token_length}, batch_size={batch_size}"
        )

    # Benchmark time placeholder
    time_model_loading_list = []
    time_processor_loading_list = []
    time_input_processing_list = []
    time_output_decoding_list = []
    time_first_token_latency_list = []
    time_generation_list = []
    time_e2e_list = []
    throughput_e2e_list = []
    throughput_generation_list = []

    # Input waveform
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
        trust_remote_code=True,
    )
    audio_sample = ds[0]["audio"]
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]

    # Make batch
    batch_waveform = [
        waveform + np.random.normal(0, np.std(waveform) * 2, len(waveform))
        for _ in range(batch_size)
    ]

    # Start benchmarking
    for _ in range(n_iter):
        if verbose_run:
            print(f"ITERATION: {_+1}/{n_iter}")

        # Run benchmark
        (
            time_model_loading,
            time_processor_loading,
            time_input_processing,
            time_output_decoding,
            time_first_token_latency,
            time_generation,
            throughput_generation,
        ) = run_model(model_name, batch_waveform, sampling_rate, output_token_length)
        time_model_loading_list.append(time_model_loading)
        time_processor_loading_list.append(time_processor_loading)
        time_input_processing_list.append(time_input_processing)
        time_output_decoding_list.append(time_output_decoding)
        time_first_token_latency_list.append(time_first_token_latency)
        time_generation_list.append(time_generation)

        # Calculate end to end time
        time_e2e = time_input_processing + time_first_token_latency + time_generation
        time_e2e_list.append(time_e2e)

        throughput_generation_list.append(throughput_generation)
        throughput_e2e = batch_size * output_token_length / time_e2e
        throughput_e2e_list.append(throughput_e2e)

        # Report
        if verbose_run:
            print(f"\tEnd to end time: {time_e2e}")
            print(f"\t\tModel loading time: {time_model_loading} seconds")
            print(f"\t\tProcessor loading time: {time_processor_loading} seconds")
            print(f"\t\tInput Processing time: {time_input_processing} seconds")
            print(f"\t\tOutput decoding time: {time_output_decoding} seconds")
            print(f"\t\tFirst token latency: {time_first_token_latency} seconds")
            print(f"\t\tGeneration time: {time_generation} seconds")
            print(f"\tThroughput (e2e): {throughput_e2e} tokens/second")
            print(f"\tThroughput (generation): {throughput_generation} tokens/second")

    # Calculate average
    time_e2e_avg = sum(time_e2e_list) / n_iter
    time_model_loading_avg = sum(time_model_loading_list) / n_iter
    time_processor_loading_avg = sum(time_processor_loading_list) / n_iter
    time_input_processing_avg = sum(time_input_processing_list) / n_iter
    time_output_decoding_avg = sum(time_output_decoding_list) / n_iter
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
        print(f"\t\tProcessor loading time: {time_processor_loading_avg} seconds")
        print(f"\t\tInput Processing time: {time_input_processing_avg} seconds")
        print(f"\t\tOutput decoding time: {time_output_decoding_avg} seconds")
        print(f"\t\tFirst token latency: {time_first_token_latency_avg} seconds")
        print(f"\t\tGeneration time: {time_generation_avg} seconds")
        print(f"\tThroughput (e2e): {throughput_e2e_avg} tokens/second")
        print(f"\tThroughput (generation): {throughput_generation_avg} tokens/second")
        print("\n\n")

    return (
        time_model_loading_avg,
        time_processor_loading_avg,
        time_input_processing_avg,
        time_output_decoding_avg,
        time_first_token_latency_avg,
        time_generation_avg,
        time_e2e_avg,
        throughput_e2e_avg,
        throughput_generation_avg,
    )


def run_all_benchmark(test_scenario):
    # Write header to CSV
    result_csv_filepath = (
        f'{test_scenario.split("/")[-1]}_perf_data_{time.strftime("%Y%m%d%H%M%S")}.csv'
    )
    write_csv_file(
        ",".join(WHISPER_PERF_BENCHMARK_OUTPUT_CSV_COLUMNS),
        result_csv_filepath,
        append=False,
    )

    # Read config
    model_name = WHISPER_PERF_BENCHMARK_CONFIG_DICT[test_scenario]["model_name"]
    output_token_length_list = WHISPER_PERF_BENCHMARK_CONFIG_DICT[test_scenario][
        "output_token_length"
    ]
    batch_size_list = WHISPER_PERF_BENCHMARK_CONFIG_DICT[test_scenario]["batch_size"]
    n_iterations = WHISPER_PERF_BENCHMARK_CONFIG_DICT[test_scenario]["n_iterations"]

    # Warm up
    run_model(model_name, np.random.normal(0, 0.5, 93680), 16000, 8)

    # Run benchmarks
    for output_token_length in output_token_length_list:
        for batch_size in batch_size_list:
            results = run_benchmark(
                model_name,
                output_token_length,
                batch_size,
                n_iterations,
            )
            write_csv_file(
                f"{model_name},{output_token_length},{batch_size},{','.join(map(str, results))}",
                result_csv_filepath,
            )


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    args = parse_args()
    run_all_benchmark(args.test_scenario)
