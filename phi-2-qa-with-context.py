import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_TOKEN_LENGTH = 2048

def read_text(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_model_output(model, tokenizer, prompt, answer_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    length = min(inputs['input_ids'].size()[1] + answer_length, MAX_TOKEN_LENGTH)
    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token, max_length=length)
    text = tokenizer.batch_decode(outputs)[0]
    return text


# Load the model and tokenizer
torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# User input question
question = "What is the difference between NC and NCC H100 VMs?"

# Load context
ncv5_post = read_text(f"{os.path.dirname(os.path.realpath(__file__))}/text/Azure_nc_h100_v5_blog_post.txt")
nccv5_post = read_text(f"{os.path.dirname(os.path.realpath(__file__))}/text/Azure_nc_h100_v5_blog_post.txt")
contexts = ncv5_post + '\n' + nccv5_post

# Answer the question without context
print(f"\nAnswer without context: {question}")
prompt = question
output = get_model_output(model, tokenizer, prompt, answer_length=300)
answer = output[len(prompt):]
print(answer)

# Answer the question with context
print(f"\nAnswer with context: {question}")
prompt = f"""Instruct:Answer based on context:\n\n{contexts}\n\n{question}\nOutput:"""
output = get_model_output(model, tokenizer, prompt, answer_length=300)
answer = output[len(prompt):]
print(answer)