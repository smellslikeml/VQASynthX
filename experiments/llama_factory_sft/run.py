import os
from llamafactory import run_exp

def main():
    # Define arguments for the SFT experiment
    # These map to the features described in LLaMA-Factory's README
    # for model, data, and training configuration.
    args = dict(
        stage="sft",                                    # Supervised Fine-Tuning
        do_train=True,
        model_name_or_path="Qwen/Qwen-VL-Chat",         # A supported VLM from the README
        
        # Dataset configuration
        dataset="vqasynth_spatial",                     # Custom dataset name
        dataset_dir="./experiments/llama_factory_sft",  # Directory containing dataset_info.json
        template="qwen",                                # Template for the Qwen model family
        
        # Finetuning method
        finetuning_type="lora",                         # Use LoRA for efficient tuning
        lora_target="all",                              # Apply LoRA to all available layers
        
        # Training parameters
        output_dir="saves/Qwen-VL-Chat/lora/sft",       # Directory to save adapters
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=1000,
        learning_rate=1e-4,
        num_train_epochs=3.0,
        plot_loss=True,
        fp16=True,                                      # Use mixed-precision training
    )
    
    # Run the experiment
    run_exp(args)

if __name__ == "__main__":
    # Ensure the output directory for plots exists
    os.makedirs("saves/Qwen-VL-Chat/lora/sft", exist_ok=True)
    main()
