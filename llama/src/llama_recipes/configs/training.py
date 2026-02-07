from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str=""
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=2048
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-5
    weight_decay: float=0.01
    gamma: float= 0.85
    seed: int=1234
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = ""
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    opt: str = "adamw"
    rho: float = 0.01
    sparsity: float = 0.5
    num_samples: int = 2
    update_freq = 5
    adaptive: bool = True
    dist_checkpoint_root_folder: str="" # will be used if using FSDP
    dist_checkpoint_folder: str="" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False 
    # log_path: str = "PATH/to/save/PEFT/log"
    data_path: str=''
    data_split: str='10,0,0'
    prefix_instruction: bool=False # whether skipping the loss of instruction
    adding_demonstrations: bool=True # whether using the prompt for instruction and input
    prompt_input: bool=False
    end_of_conversation_token: str="<|endoftext|>"
    template_path: str=""
    recipe: str=""
    cache_dir: str=""
    data_output_path: str=""
    local_rank: int=0
    fine_tuning_mse: bool=False
    pairwise_allresponse: bool=False
    preprocessing_num_workers: int=16
