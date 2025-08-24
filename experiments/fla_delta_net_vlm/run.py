import torch
from transformers import AutoModelForCausalLM, AutoConfig
from fla.models.delta_net.configuration_delta_net import DeltaNetConfig
from fla.models.delta_net.modeling_delta_net import DeltaNetAttention
import warnings

# The VQASynth project fine-tunes VLMs, which often use Llama-like architectures.
# This script demonstrates how to replace the standard quadratic-time attention mechanism
# in such a model with a highly efficient linear-time alternative, DeltaNet,
# from the flash-linear-attention library. This can significantly speed up training
# and reduce memory usage, especially for long sequences common in CoT reasoning.


def convert_llama_config_to_delta_net_config(
    llama_config: AutoConfig,
) -> DeltaNetConfig:
    """
    Maps attributes from a LlamaConfig to a DeltaNetConfig to ensure compatibility.
    """
    return DeltaNetConfig(
        vocab_size=llama_config.vocab_size,
        hidden_size=llama_config.hidden_size,
        intermediate_size=llama_config.intermediate_size,
        num_hidden_layers=llama_config.num_hidden_layers,
        num_attention_heads=llama_config.num_attention_heads,
        num_key_value_heads=getattr(
            llama_config, "num_key_value_heads", llama_config.num_attention_heads
        ),
        hidden_act=llama_config.hidden_act,
        max_position_embeddings=llama_config.max_position_embeddings,
        initializer_range=llama_config.initializer_range,
        rms_norm_eps=llama_config.rms_norm_eps,
        use_cache=False,  # Caching is not implemented in this minimal example
        tie_word_embeddings=llama_config.tie_word_embeddings,
    )


def patch_model_with_delta_net(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Iterates through the model's layers and replaces LlamaAttention modules
    with DeltaNetAttention modules, copying weights where possible.
    """
    for i, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        llama_config = original_attn.config

        # Create a compatible config for DeltaNet
        delta_net_config = convert_llama_config_to_delta_net_config(llama_config)

        # Instantiate the new attention mechanism from fla
        new_attn = DeltaNetAttention(config=delta_net_config, layer_idx=i)

        # Copy the projection weights from the original attention layer
        new_attn.q_proj.weight = original_attn.q_proj.weight
        new_attn.k_proj.weight = original_attn.k_proj.weight
        new_attn.v_proj.weight = original_attn.v_proj.weight
        new_attn.o_proj.weight = original_attn.o_proj.weight

        # Replace the module in the model
        layer.self_attn = new_attn

    # Disable cache usage in the model's main config as our wrapper doesn't support it
    model.config.use_cache = False
    warnings.warn(
        "Model patched with DeltaNet. `use_cache` is disabled as it's not supported in this experiment."
    )

    return model


def main():
    """
    Main function to run the experiment.
    """
    # Use a small, accessible Llama-based model for the demonstration
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print(
            "Warning: CUDA not available. FLA/Triton are optimized for GPUs. The script will run on CPU but may be slow."
        )
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    print(f"Loading original model '{model_id}' to device '{device}'...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)

    print("\nPatching LlamaAttention with fla.models.DeltaNetAttention...")
    model = patch_model_with_delta_net(model)
    print("Patching complete.")
    print("\nPatched model architecture (showing a decoder layer):")
    print(model.model.layers[0])

    # Prepare dummy data for a test training step
    batch_size = 2
    seq_len = 256
    inputs = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)
    labels = inputs.clone()

    print(f"\nTesting forward pass with input shape: {inputs.shape}...")
    try:
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        print(f"Forward pass successful. Loss: {loss.item():.4f}")

        print("Testing backward pass...")
        loss.backward()
        print("Backward pass successful.")

        print(
            "\n✅ Experiment successful: Model patched with DeltaNet and completes a training step."
        )

    except Exception as e:
        print(f"\n❌ Experiment failed during forward/backward pass.")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
