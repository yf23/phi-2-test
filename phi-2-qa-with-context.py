import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def read_text(file_path):
    with open(file_path, "r") as file:
        return file.read()


torch.set_default_device("cuda")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# User input question
question = "What is the difference between Azure NCv5 and NCCv5?"

# Load context
ncv5_post = read_text("text\Azure_nc_h100_v5_blog_post.txt")
nccv5_post = read_text("text\Azure_nc_h100_v5_blog_post.txt")
contexts = ncv5_post + '\n' + nccv5_post
print(contexts)

# Answer the question without context
inputs = tokenizer(question, return_tensors="pt", return_attention_mask=False)
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]

# Answer the question with context
prompt = """Instruct:Answer based on context:\n\n{contexts}\n\n{question}.\nOutput:"""
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]