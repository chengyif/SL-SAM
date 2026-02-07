import os
import shutil
from transformers import AutoModelForCausalLM, LlamaTokenizer

source_model_path = ""
source_tokenizer_path = ""
target_path = ""

os.makedirs(target_path, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(source_model_path)
tokenizer = LlamaTokenizer.from_pretrained(source_tokenizer_path)
model.save_pretrained(target_path)
tokenizer.save_pretrained(target_path)
