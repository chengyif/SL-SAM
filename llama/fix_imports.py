#!/usr/bin/env python3

import os
import re
import glob

ft_datasets_dir = ''

old_import_pattern = re.compile(r'from chat_utils import (.*)')

new_import_template = 'from llama_recipes.inference.chat_utils import \1'

incomplete_import_pattern = re.compile(r'from llama_recipes\.inference\.chat_utils import\s+\n')
new_complete_import = 'from llama_recipes.inference.chat_utils import format_conv, format_tokens\n'

utils_import_pattern = re.compile(r'from utils import (.*)')

new_utils_import_template = 'from llama_recipes.ft_datasets.utils import \1'

old_path_pattern = re.compile(r"sys\.path\.insert\(1, 'inference/'\)\s*sys\.path\.insert\(1, 'utils/'\)")


new_path_code = '''

'''

python_files = glob.glob(os.path.join(ft_datasets_dir, '*.py'))

for file_path in python_files:
    if os.path.basename(file_path) == '__init__.py':
        continue
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    modified_content = old_import_pattern.sub(new_import_template, content)
    modified_content = incomplete_import_pattern.sub(new_complete_import, modified_content)
    modified_content = utils_import_pattern.sub(new_utils_import_template, modified_content)
    modified_content = old_path_pattern.sub(new_path_code, modified_content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)