LLM_PERF_BENCHMARK_CONFIG_DICT = {
    "phi-2": {
        "model_name": "microsoft/phi-2",
        "input_token_length": [32, 256, 1024, 2048],
        "output_token_length": [32, 64, 128],
        "batch_size": [1, 8, 32, 64],
        "n_iterations": 10,
    },
    "gpt2-xl": {
        "model_name": "openai-community/gpt2-xl",
        "input_token_length": [32, 256, 512],
        "output_token_length": [32, 64, 256],
        "batch_size": [1, 8, 32, 64],
        "n_iterations": 10,
    },
    "phi-2-test": {
        "model_name": "microsoft/phi-2",
        "input_token_length": [2048],
        "output_token_length": [128],
        "batch_size": [64],
        "n_iterations": 1,
    },
}

LLM_PERF_BENCHMARK_OUTPUT_CSV_COLUMNS = [
    "model_name",
    "input_token_length",
    "output_token_length",
    "batch_size",
    "model_loading_latency",
    "tokenizer_loading_latency",
    "input_tokenization_latency",
    "output_decoding_latency",
    "first_token_latency",
    "generation_latency",
    "total_latency",
    "throughput_e2e",
    "throughput_generation",
]

WHISPER_PERF_BENCHMARK_CONFIG_DICT = {
    "whisper-large-v3": {
        "model_name": "openai/whisper-large-v3",
        "output_token_length": [8, 32, 64, 128, 256, 444],
        "batch_size": [1, 8, 32, 64, 128, 192],
        "n_iterations": 5,
    },
    "whisper-large-v3-test": {
        "model_name": "openai/whisper-large-v3",
        "output_token_length": [444],
        "batch_size": [2],
        "n_iterations": 1,
    },
}

WHISPER_PERF_BENCHMARK_OUTPUT_CSV_COLUMNS = [
    "model_name",
    "output_token_length",
    "batch_size",
    "model_loading_latency",
    "processsor_loading_latency",
    "input_processing_latency",
    "output_decoding_latency",
    "first_token_latency",
    "generation_latency",
    "total_latency",
    "throughput_e2e",
    "throughput_generation",
]
