"""
The file for creating dataset for training and evaluation.
Supported datasets:
- Alpaca
- ShareGPT52K
- MetaMathQA
- Open-Platypus
- Auto-Wiki

Part of the code was adopted from https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/data/data_utils.
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
from deepspeed.accelerator import get_accelerator

from datasets import load_dataset, load_from_disk


# ------------------------------------------------------------
# 1. PromptRawDataset
# ------------------------------------------------------------
# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if os.path.exists(dataset_name):
            self.raw_datasets = load_from_disk(dataset_name)
        elif not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        if hasattr(self.raw_datasets, "train"):
            return self.raw_datasets["train"]
        return self.raw_datasets

    def get_eval_data(self):
        if hasattr(self.raw_datasets, "test"):
            return self.raw_datasets["test"]
        elif hasattr(self.raw_datasets, "validation"):
            return self.raw_datasets["validation"]
        return self.raw_datasets

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return None

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return

class AlpacaDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        self.dataset_name = "tatsu-lab/alpaca"
        self.dataset_name_clean = "tatsu_lab_alpaca"
        super().__init__(output_path, seed, local_rank, dataset_name="tatsu-lab/alpaca")

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset
    
    def get_prompt_and_chosen(self, sample):

        if sample["input"] != "":
            return "[INST] " + sample["instruction"] +" "+ sample["input"] + " [/INST] " + " " + sample["output"]
        else:
            return "[INST] " + sample["instruction"] + " [/INST] " + " " + sample["output"]
    
    def get_prompt_and_rejected(self, sample):
        return None

class ShareGPT52KDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        self.dataset_name = "LNTANOooo/sharegpt52k"
        self.dataset_name_clean = "LNTANOooo_sharegpt52k"
        super().__init__(output_path, seed, local_rank, dataset_name="LNTANOooo/sharegpt52k")

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        # index = get_raw_dataset_split_index(self.local_rank, self.output_path,
        #                                     self.dataset_name_clean,
        #                                     self.seed, "train_eval", "1,9", 0,
        #                                     len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    # def get_prompt_and_chosen(self, sample):
    #     text = ""
    #     for conv in sample["conversation"]:
    #         role = conv["role"]
    #         content = conv["content"]
    #         if type(content) == str:
    #             if role == "user":
    #                 text += "[INST] " + content + " [/INST] "
    #             elif role == "assistant":
    #                 text += content + " "
    #             else:
    #                 break
    #         else:
    #             break
    #     return text

    def get_prompt_and_chosen(self, sample):
        text = "<|begin_of_text|>"
        for conv in sample["conversation"]:
            role = conv["role"]
            content = conv["content"]
            if type(content) == str:
                if role == "user":
                    text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "assistant":
                    text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
                else:
                    break
            else:
                break
        return text

    def get_prompt_and_rejected(self, sample):
        return None


class MetaMathQA(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        self.dataset_name = "meta-math/MetaMathQA"
        self.dataset_name_clean = "meta_math_meta_math_qa"
        super().__init__(output_path, seed, local_rank, dataset_name="meta-math/MetaMathQA")

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset
    
    def get_prompt_and_chosen(self, sample):
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample["query"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{sample["response"]}<|eot_id|>"""

    def get_prompt_and_rejected(self, sample):
        return None

class OpenPlatypusDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        self.dataset_name = "garage-bAInd/Open-Platypus"
        self.dataset_name_clean = "garage_b_a_ind_open_platypus"
        super().__init__(output_path, seed, local_rank, dataset_name="garage-bAInd/Open-Platypus")

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        # index = get_raw_dataset_split_index(self.local_rank, self.output_path,
        #                                     self.dataset_name_clean,
        #                                     self.seed, "train_eval", "1,9", 0,
        #                                     len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset
    
    # def get_prompt_and_chosen(self, sample):
    #     return "[INST] " + sample["instruction"] + " [/INST] " + " " + sample["output"]

    def get_prompt_and_chosen(self, sample):
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample["instruction"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{sample["output"]}<|eot_id|>"""

    def get_prompt_and_rejected(self, sample):
        return None

class AutoWikiDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        self.dataset_name = "chaojiang06/wiki_auto"
        self.dataset_name_clean = "chaojiang06_wiki_auto"
        super().__init__(output_path, seed, local_rank, dataset_name="chaojiang06/wiki_auto")

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt_and_chosen(self, sample):
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample["instruction"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{sample["output"]}<|eot_id|>"""

    def get_prompt_and_rejected(self, sample):
        return None

class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids":
                self.chosen_dataset[idx]["input_ids"],
                "attention_mask":
                self.chosen_dataset[idx]["attention_mask"],
                "labels":
                torch.where(self.chosen_dataset[idx]["attention_mask"].bool(),
                            self.chosen_dataset[idx]["input_ids"], -100)
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id


def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    if dataset_name == "Alpaca":
        return AlpacaDataset(output_path, seed, local_rank)
    elif dataset_name == "ShareGPT52K":
        return ShareGPT52KDataset(output_path, seed, local_rank)
    elif dataset_name == "MetaMathQA":
        return MetaMathQA(output_path, seed, local_rank)
    elif dataset_name == "OpenPlatypus":
        return OpenPlatypusDataset(output_path, seed, local_rank)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank,
                                output_path,
                                dataset_name,
                                seed,
                                split_name,
                                data_split,
                                split_index,
                                data_size,
                                rebuild=False):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    # reindex each time when using local jsonfile since it's more likely to get modified
    if rebuild or (not os.path.isfile(index_file_name)) or (dataset_name
                                                            == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_dataset.append(chosen_token)
                reject_dataset.append(reject_token)
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    elif train_phase == 3:
        filtered = 0
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                if prompt_token["input_ids"].size()[-1] <= max_seq_len:
                    for key_word in ["input_ids", "attention_mask"]:
                        prompt_token[key_word] = prompt_token[
                            key_word].squeeze(0).flip(0)
                    prompt_dataset.append(prompt_token)
                else:
                    filtered += 1
        print(f'Creating dataset {raw_dataset.dataset_name_clean} '
              f'for {train_phase=} size={len(prompt_dataset)} {filtered=}')

    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, rebuild):
    print(f"Creating dataset {dataset_name} for {train_phase=}")
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset), rebuild)
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset), rebuild)
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset

def create_prompt_dataset(
        train_config, train_phase, tokenizer, end_of_conversation_token):
    """
    Creates the prompt dataset
    """
    local_rank = train_config.local_rank
    data_path = train_config.data_path
    data_split = train_config.data_split
    output_path = train_config.data_output_path
    seed = train_config.seed
    max_seq_len = train_config.context_length
    sft_only_data_path = []
    reload = False

    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    # buf_create_cache = torch.ByteTensor([not cache_found]).to(
    #     get_accelerator().current_device_name())
    # torch.distributed.all_reduce(buf_create_cache)

    print(data_path)
    print(len(data_path))

    # if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
    print(f'Creating prompt dataset {data_path}, {reload=}')
    if len(data_path) == 1:  # Single dataset.
        print("Creating dataset for single dataset")
        train_dataset, eval_dataset = create_dataset(
            local_rank,
            data_path[0],
            data_split,
            output_path,
            train_phase,
            seed,
            tokenizer,
            end_of_conversation_token,
            max_seq_len,
            rebuild=reload)
    else:  # Blending datasets.
        train_datasets = []
        eval_datasets = []
        train_size = 0
        eval_size = 0
        for d_path in data_path:
            train_dataset, eval_dataset = create_dataset(
                local_rank,
                [d_path],
                data_split,
                output_path,
                train_phase,
                seed,
                tokenizer,
                end_of_conversation_token,
                max_seq_len,
                rebuild=reload)
            train_datasets.append(train_dataset)
            eval_datasets.append(eval_dataset)
            train_size += len(train_dataset)
            eval_size += len(eval_dataset)
        train_dataset = ConcatDataset(train_datasets)
        shuffle_idx = get_shuffle_idx(seed, train_size)
        train_dataset = Subset(train_dataset, shuffle_idx.tolist())
        eval_dataset = ConcatDataset(eval_datasets)
        shuffle_idx = get_shuffle_idx(seed, eval_size)
        eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

    # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
    if train_phase == 1 and sft_only_data_path:
        sft_train_datasets = []
        sft_eval_datasets = []
        sft_train_size = 0
        sft_eval_size = 0
        for sft_path in sft_only_data_path:
            sft_train_dataset, sft_eval_dataset = create_dataset(
                local_rank,
                sft_path,
                "10,0,0",
                output_path,
                train_phase,
                seed,
                tokenizer,
                end_of_conversation_token,
                max_seq_len,
                rebuild=reload)
            sft_train_datasets.append(sft_train_dataset)
            sft_eval_datasets.append(sft_eval_dataset)
            sft_train_size += len(sft_train_dataset)
            sft_eval_size += len(sft_eval_dataset)
        if sft_train_datasets:  # Check if sft_train_datasets is not empty
            sft_train_dataset = ConcatDataset(sft_train_datasets)
            train_dataset = ConcatDataset(
                [train_dataset, sft_train_dataset])
            shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
        if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
            sft_eval_dataset = ConcatDataset(sft_eval_datasets)
            eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
            shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
    torch.save(train_dataset, train_fname)
    torch.save(eval_dataset, eval_fname)
    # torch.distributed.barrier()
    return torch.load(train_fname,
                      weights_only=False), torch.load(eval_fname,
                                                      weights_only=False)
