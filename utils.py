import time
from transformers import TextStreamer

BaseStreamer = TextStreamer.__bases__[0]


class ThroughputStreamer(BaseStreamer):
    def __init__(self):
        self.first_token_time = -1
        self.last_token_time = -1
        self.skip_prompt = True
        self.n_tokens = 0
        self.tot_latency = -1
        self.ftl = -1

    def put(self, value):
        # Prompt tokens sent first, ignore
        if self.skip_prompt:
            self.skip_prompt = False
            return

        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter()

        self.n_tokens += len(value.flatten())

    def end(self):
        self.last_token_time = time.perf_counter()

    def generation_latency(self):
        return self.last_token_time - self.first_token_time

    def set_latencies(self, pre_generate_start_time, post_generate_end_time):
        self.ftl = self.first_token_time - pre_generate_start_time
        self.tot_latency = post_generate_end_time - pre_generate_start_time

    def first_token_latency(self):
        return self.ftl

    def throughput(self):
        return self.n_tokens / self.generation_latency()

    def total_latency(self):
        return self.tot_latency


def write_csv_file(line, filepath, append=True):
    mode = "a" if append else "w"
    with open(filepath, mode) as f:
        f.write(line + "\n")
