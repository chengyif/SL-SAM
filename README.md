# This is the code for the paper "Sparse Layer Sharpness-Aware Minimization for Efficient Fine-Tuning"
# We provide the code for running the experiments of fine-tuning ViT, RoBERTa and Llama-3.2-3B-Instruct models.

# If fine-tune the ViT-Small model with the SL-SAM optimizer
cd vit
python train.py --lr 1e-4 --opt slsam --s 0.2 --epochs 20

# If fine-tune the RoBERTa model on the mnli dataset with the SL-SAM optimizer
cd roberta
python run_glue.py \
    --model_name_or_path ...
    --task_name mnli \
    --max_length 512 \
    --seed=1234 \
    --per_device_train_batch_size 16 \
    --opt slsam \
    --learning_rate 1e-5 \
    --rho 0.01 \
    --s 0.2 \
    --num_train_epochs 30 

# If fine-tune Llama model with the SL-SAM optimizer
cd llama
python src/finetuning2.py --dist_checkpoint_root_folder save_dir --dist_checkpoint_folder llama --use_fast_kernels False --batch_size_training 2 \
--num_epochs 1 --opt slsam --lr 1e-5 --rho 0.01 --sparsity 0.2 --num_samples 1 --adaptive False --data_split 10,0,0 --prefix_instruction True \
--adding_demonstrations False --prompt_input True --end_of_conversation_token '<|eot_id|>' --template_path 123 --recipe 123 --peft_method None --data_path ['OpenPlatypus']