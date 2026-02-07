#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import torch
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description='Llama')
    parser.add_argument('--model_path', type=str, 
                        default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_text', type=str)
    parser.add_argument('--max_length', type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        device_map='auto' if args.device == 'cuda' else None
    )

    inputs = tokenizer(args.test_text, return_tensors="pt")
    if args.device == 'cuda':
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()