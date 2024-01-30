import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_TOKEN_LENGTH = 2048

def read_text(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_model_output(model, tokenizer, prompt, answer_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    length = min(inputs['input_ids'].size()[1] + answer_length, MAX_TOKEN_LENGTH)
    outputs = model.generate(**inputs, max_length=length)
    text = tokenizer.batch_decode(outputs)[0]
    return text


# Load the model and tokenizer
torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# User input question
question = "What is the difference between Azure NCv5 and NCCv5?"

# Load context
ncv5_post = read_text("./text/Azure_nc_h100_v5_blog_post.txt")
nccv5_post = read_text("./text/Azure_nc_h100_v5_blog_post.txt")
contexts = ncv5_post + '\n' + nccv5_post

# Answer the question without context
print("Answer without context:")
prompt = question
output = get_model_output(model, tokenizer, prompt, answer_length=200)
answer = output[len(prompt):]
print(answer)

# Answer the question with context
print("Answer with context:")
prompt = f"""Instruct:Answer based on context:\n\n{contexts}\n\n{question}\nOutput:"""
output = get_model_output(model, tokenizer, prompt, answer_length=200)
answer = output[len(prompt):]
print(answer)