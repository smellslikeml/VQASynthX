import argparse
import copy
import gc
import numpy as np
import torch
import optuna
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Attempt to import LLaVA components, assuming they are in the environment
try:
    from llava.model import LlavaLlamaForCausalLM
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
except ImportError:
    print("Warning: LLaVA components not found. This script is designed for LLaVA models.")
    LlavaLlamaForCausalLM = None
    load_pretrained_model = None
    get_model_name_from_path = None


def find_layers(module, layers=[torch.nn.Linear], name=''):
    """Recursively find all layers of a certain type in a module."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name_child, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name_child if name != '' else name_child
        ))
    return res

def get_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """Generates synthetic calibration data."""
    print(f"Generating {n_samples} calibration samples...")
    # Generate random token IDs for calibration
    # In a real scenario, this should come from a representative dataset like C4 or Wikitext
    calib_data = torch.randint(0, tokenizer.vocab_size, (n_samples, seq_len))
    return [calib_data[i, :] for i in range(n_samples)]

class WandaPruner:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layers = find_layers(model)
        self.activations = {}

    @torch.no_grad()
    def collect_activations(self, calibration_data):
        """Collects activation magnitudes using calibration data."""
        print("Collecting activations...")
        for name, layer in self.layers.items():
            self.activations[name] = torch.zeros(layer.in_features, device=self.device)

        hooks = []
        for name, layer in self.layers.items():
            def hook_factory(name):
                def hook(module, inp, out):
                    # Use the first element of input tuple
                    inp_tensor = inp[0].float()
                    # Accumulate sum of squared activations
                    self.activations[name] += torch.sum(inp_tensor.abs().pow(2), dim=[0, 1])
                return hook
            hooks.append(layer.register_forward_hook(hook_factory(name)))

        for data in tqdm(calibration_data):
            self.model(data.unsqueeze(0).to(self.device))

        for hook in hooks:
            hook.remove()
        
        # Normalize activations
        for name in self.activations:
            self.activations[name] = torch.sqrt(self.activations[name])

    @torch.no_grad()
    def prune(self, sparsity_ratios):
        """Prunes the model based on weight-activation product (Wanda)."""
        for name, layer in self.layers.items():
            sparsity = sparsity_ratios.get(name, 0.0)
            if sparsity <= 0:
                continue

            W = layer.weight.data.clone().float()
            if name not in self.activations:
                print(f"Warning: Activations for layer {name} not found. Skipping.")
                continue
            
            act = self.activations[name].unsqueeze(0)
            
            # Compute Wanda metric
            wanda_metric = (W.abs() * act).cpu()

            # Determine threshold for pruning
            num_to_prune = int(wanda_metric.numel() * sparsity)
            if num_to_prune == 0:
                continue
                
            threshold = torch.sort(wanda_metric.flatten())[0][num_to_prune].to(self.device)

            # Create and apply mask
            mask = wanda_metric >= threshold
            layer.weight.data *= mask.to(self.device)


def eval_perplexity(model, tokenizer, device):
    """Evaluates perplexity on a small synthetic dataset as a proxy for performance."""
    # Using a small, fixed synthetic dataset for demonstration
    test_data = get_calibration_data(tokenizer, n_samples=32, seq_len=1024)
    
    nlls = []
    with torch.no_grad():
        for tokens in tqdm(test_data, desc="Evaluating Perplexity"):
            tokens = tokens.unsqueeze(0).to(device)
            outputs = model(tokens, labels=tokens)
            nlls.append(outputs.loss.item())
    
    ppl = np.exp(np.mean(nlls))
    print(f"Perplexity: {ppl:.4f}")
    return ppl

def check_sparsity(model):
    total_params = 0
    total_zeros = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total_params += module.weight.nelement()
            total_zeros += torch.sum(module.weight == 0)
    return total_zeros / total_params if total_params > 0 else 0

def objective(trial, base_model, tokenizer, calibration_data, device):
    """Objective function for Optuna to minimize."""
    # Create a fresh copy of the model for each trial
    model_copy = copy.deepcopy(base_model)
    pruner = WandaPruner(model_copy, tokenizer, device)
    pruner.collect_activations(calibration_data)

    # Define search space for sparsity ratios for each layer
    sparsity_ratios = {}
    for name in pruner.layers.keys():
        # Suggest a uniform sparsity ratio for demonstration
        # A more advanced approach would group layers by type or depth
        sparsity_ratios[name] = trial.suggest_float(f"{name}_sparsity", 0.2, 0.8)
    
    # Prune the model copy
    pruner.prune(sparsity_ratios)

    # Evaluate the performance (lower perplexity is better)
    ppl = eval_perplexity(model_copy, tokenizer, device)

    # Clean up GPU memory
    del model_copy, pruner
    gc.collect()
    torch.cuda.empty_cache()

    return ppl

def main():
    parser = argparse.ArgumentParser(description="Adaptive sparsity pruning using Optuna, inspired by OptSpa.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the LLaVA model, e.g., "liuhaotian/llava-v1.5-7b"')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cache HuggingFace models.')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials to run.')
    parser.add_argument('--save_model_path', type=str, default="./pruned_model", help='Path to save the final pruned model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    if LlavaLlamaForCausalLM is None:
        raise RuntimeError("LLaVA is not installed. Please install it to proceed.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(args.model_path, None, model_name, device_map='auto', cache_dir=args.cache_dir)
    
    # Get calibration data
    calibration_data = get_calibration_data(tokenizer)

    # Create and run Optuna study
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(
        lambda trial: objective(trial, model, tokenizer, calibration_data, device),
        n_trials=args.n_trials
    )

    print(f"Optuna study finished. Best PPL: {study.best_value:.4f}")
    best_sparsity_ratios = study.best_params

    # Apply the best found sparsity ratios to the original model
    print("Applying best sparsity configuration to the model...")
    final_pruner = WandaPruner(model, tokenizer, device)
    final_pruner.collect_activations(calibration_data)
    final_pruner.prune({k.replace('_sparsity', ''): v for k, v in best_sparsity_ratios.items()})

    final_sparsity = check_sparsity(model)
    print(f"Final model sparsity: {final_sparsity:.4f}")

    # Save the pruned model
    if args.save_model_path:
        print(f"Saving pruned model to {args.save_model_path}...")
        model.save_pretrained(args.save_model_path)
        tokenizer.save_pretrained(args.save_model_path)

    print("Experiment complete.")

if __name__ == '__main__':
    main()
