import argparse
import os
import pandas as pd
import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm


# This model architecture is a direct adaptation of the BloomBERT model
# found in the source repository at `src/model/bloombert.py`.
# It uses a pre-trained DistilBERT model with a sequence classification head.
class BloomBERTClassifier(nn.Module):
    def __init__(self, output_dim):
        super(BloomBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def predict_batch(texts, model, tokenizer, device, batch_size=32):
    model.eval()
    predictions = []
    category_map = {
        0: "Remember",
        1: "Understand",
        2: "Apply",
        3: "Analyse",
        4: "Evaluate",
        5: "Create",
    }

    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying Questions"):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=128,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend([category_map[p] for p in preds])

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VQA questions for cognitive complexity using a BloomBERT-like model."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input parquet file containing generated questions.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output parquet file with complexity annotations.",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default="question",
        help="The name of the column containing the questions to analyze.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference."
    )

    args = parser.parse_args()

    print(f"Loading data from {args.input_path}...")
    df = pd.read_parquet(args.input_path)

    if args.question_column not in df.columns:
        raise ValueError(
            f"Question column '{args.question_column}' not found in the input data."
        )

    # Set up model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NOTE: In a production scenario, the trained weights for BloomBERT would be loaded here.
    # For this minimal, testable feature, we use the base DistilBERT weights.
    # The classification head is randomly initialized, so the predictions demonstrate
    # the pipeline integration rather than accurate classification.
    num_labels = 6  # (Remember, Understand, Apply, Analyse, Evaluate, Create)
    model = BloomBERTClassifier(output_dim=num_labels).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Get texts to classify
    texts_to_classify = df[args.question_column].astype(str).tolist()

    # Run prediction
    bloom_levels = predict_batch(
        texts_to_classify, model, tokenizer, device, args.batch_size
    )

    # Add results to dataframe
    df["bloom_complexity_level"] = bloom_levels

    print(f"Complexity analysis complete. Example distribution:")
    print(df["bloom_complexity_level"].value_counts())

    # Save updated dataframe
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"Successfully saved annotated data to {args.output_path}")


if __name__ == "__main__":
    main()
