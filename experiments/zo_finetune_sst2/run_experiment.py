import argparse
import torch
import logging
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def apply_template(examples, tokenizer):
    """Applies the SST-2 template from DPZO.sh to a batch of examples."""
    # TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
    # MAPPING="{'0':'terrible','1':'great'}"
    template = "{sent} It was {mask}."
    inputs = [
        template.format(sent=sent, mask=tokenizer.mask_token)
        for sent in examples["sentence"]
    ]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=128)


def get_verbalizer_ids(tokenizer):
    """Gets token ids for the verbalizer words."""
    # MAPPING="{'0':'terrible','1':'great'}"
    try:
        terrible_id = tokenizer.convert_tokens_to_ids("terrible")
        great_id = tokenizer.convert_tokens_to_ids("great")
        return {"0": terrible_id, "1": great_id}
    except Exception:
        logging.error(
            "Verbalizer words 'terrible' or 'great' not in tokenizer vocabulary."
        )
        raise


def zo_step(model, inputs, labels, verbalizer_ids, lr, zo_eps, device):
    """Performs a single Zeroth-Order optimization step."""

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logging.warning("No trainable parameters found. Skipping step.")
        return torch.tensor(0.0)

    # 1. Store original parameters
    original_params = [p.clone().detach() for p in trainable_params]

    # 2. Generate a random perturbation vector `z` for all trainable parameters
    perturbation = [torch.randn_like(p) for p in trainable_params]

    # --- Forward pass for positive perturbation ---
    # 3. Add perturbation: theta_plus = theta + zo_eps * z
    for p, z in zip(trainable_params, perturbation):
        p.data.add_(z, alpha=zo_eps)

    # 4. Calculate loss_plus
    outputs_plus = model(**inputs)
    mask_token_index = (inputs["input_ids"] == model.config.mask_token_id).nonzero(
        as_tuple=True
    )[1]
    masked_logits_plus = outputs_plus.logits[
        torch.arange(len(outputs_plus.logits)), mask_token_index
    ]

    verbalizer_logits_plus = masked_logits_plus[
        :, [verbalizer_ids["0"], verbalizer_ids["1"]]
    ]
    loss_fct = torch.nn.CrossEntropyLoss()
    loss_plus = loss_fct(verbalizer_logits_plus, labels)

    # --- Forward pass for negative perturbation ---
    # 5. Subtract perturbation: theta_minus = theta - 2 * zo_eps * z (from theta_plus)
    for p, z in zip(trainable_params, perturbation):
        p.data.add_(z, alpha=-2 * zo_eps)

    # 6. Calculate loss_minus
    outputs_minus = model(**inputs)
    masked_logits_minus = outputs_minus.logits[
        torch.arange(len(outputs_minus.logits)), mask_token_index
    ]

    verbalizer_logits_minus = masked_logits_minus[
        :, [verbalizer_ids["0"], verbalizer_ids["1"]]
    ]
    loss_minus = loss_fct(verbalizer_logits_minus, labels)

    # 7. Restore original parameters before update
    for p, orig_p in zip(trainable_params, original_params):
        p.data = orig_p

    # 8. Estimate gradient and update parameters
    # g_hat = (loss_plus - loss_minus) / (2 * zo_eps) * z
    # theta = theta - lr * g_hat
    grad_est_scale = (loss_plus.item() - loss_minus.item()) / (2 * zo_eps)

    with torch.no_grad():
        for p, z in zip(trainable_params, perturbation):
            p.add_(z, alpha=-lr * grad_est_scale)

    return (loss_plus + loss_minus) / 2


def evaluate(model, dataloader, verbalizer_ids, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**inputs)
            mask_token_index = (
                inputs["input_ids"] == model.config.mask_token_id
            ).nonzero(as_tuple=True)[1]
            masked_logits = outputs.logits[
                torch.arange(len(outputs.logits)), mask_token_index
            ]

            verbalizer_logits = masked_logits[
                :, [verbalizer_ids["0"], verbalizer_ids["1"]]
            ]
            preds = torch.argmax(verbalizer_logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    return total_correct / total_samples


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path).to(device)

    # Freeze all parameters except the classification head (lm_head)
    # This is a common PEFT strategy that ZO methods can make more efficient
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_trainable}")

    verbalizer_ids = get_verbalizer_ids(tokenizer)

    # Load and preprocess dataset
    dataset = load_dataset("glue", "sst2")
    tokenized_dataset = dataset.map(
        lambda x: apply_template(x, tokenizer), batched=True
    )
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["validation"], batch_size=args.per_device_eval_batch_size
    )

    logging.info("Starting Zeroth-Order Fine-tuning...")
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        for step, batch in enumerate(pbar):
            labels = batch.pop("label").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            loss = zo_step(
                model,
                inputs,
                labels,
                verbalizer_ids,
                args.learning_rate,
                args.zo_eps,
                device,
            )

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (step + 1)})

    logging.info("Training finished. Starting evaluation.")
    accuracy = evaluate(model, eval_dataloader, verbalizer_ids, device)
    logging.info(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zeroth-Order Fine-tuning Experiment inspired by DPZOPO."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="roberta-base",
        help="Model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the ZO optimizer.",
    )
    parser.add_argument(
        "--zo_eps",
        type=float,
        default=1e-3,
        help="Perturbation magnitude for ZO gradient estimation.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    main(args)
