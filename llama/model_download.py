#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description='Hugging Face')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--access_token', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='causal_lm',
                        choices=['causal_lm', 'base', 'tokenizer_only'])
    parser.add_argument('--use_auth_token', action='store_true')
    parser.add_argument('--force_download', action='store_true')
    parser.add_argument('--local_files_only', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    token = args.access_token
    if args.use_auth_token and not token:
        token_path = os.path.expanduser('~/.huggingface/token')
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                token = f.read().strip()
    
    download_kwargs = {
        'local_files_only': args.local_files_only,
        'force_download': args.force_download,
    }
    
    if token:
        download_kwargs['token'] = token
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.output_dir,
        **download_kwargs
    )
    tokenizer.save_pretrained(args.output_dir)
    
    if args.model_type != 'tokenizer_only':
        if args.model_type == 'causal_lm':
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                cache_dir=args.output_dir,
                **download_kwargs
            )
        else:  # base model
            model = AutoModel.from_pretrained(
                args.model_name,
                cache_dir=args.output_dir,
                **download_kwargs
            )
        model.save_pretrained(args.output_dir)
if __name__ == "__main__":
    main()