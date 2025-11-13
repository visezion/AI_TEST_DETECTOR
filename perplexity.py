import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

tokenizer = AutoTokenizer.from_pretrained(config.LM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(config.LM_MODEL_NAME)


def get_perplexity(text):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()
