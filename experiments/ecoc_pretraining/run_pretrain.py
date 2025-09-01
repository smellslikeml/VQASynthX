# This script integrates the core idea from the "Automated Codebook Learning with ECOC"
# paper (github.com/YuChou20/Automated-Codebook-Learning-with-Error-Correcting-Output-Code-Technique)
# into the experimental-vqasynth repository.
#
# The original code was split across multiple files (run.py, simclr.py, models/resnet_ecoc_simclr.py,
# data_aug/contrastive_learning_dataset.py, lars.py, utils.py).
#
# For a minimal, self-contained, and testable feature branch, we have consolidated the necessary
# components for the pre-training phase into this single file. This script focuses on the
# "Automated Codebook Learning" (ACL) pre-training, which uses contrastive learning along with
# a novel "Column Separation Loss" to learn a robust codebook for multi-class classification.
#
# This experiment serves as a foundational step to explore if such learned, robust representations
# can benefit downstream tasks in the VQASynth pipeline, particularly those involving
# object classification or recognition, by making them more resilient to noise or adversarial perturbations.

import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm


# ==========================================================================================
# Component from: data_aug/contrastive_learning_dataset.py and data_aug/view_generator.py
# ==========================================================================================
class ContrastiveLearningDataset:
    def __init__(self, root_folder, n_views=2):
        self.root_folder = root_folder
        self.n_views = n_views

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    def get_dataset(self, name, n_views):
        if name == "cifar10":
            transform = self.get_simclr_pipeline_transform(32)
            dataset = CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(transform, n_views),
                download=True,
            )
            return dataset
        else:
            raise RuntimeError(f"Dataset {name} not supported.")


class ContrastiveLearningViewGenerator:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


# ==========================================================================================
# Component from: lars.py
# ==========================================================================================
class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue
                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.linalg.norm(p)
                    update_norm = torch.linalg.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])


# ==========================================================================================
# Component from: models/resnet_simclr.py and models/resnet_ecoc_simclr.py
# ==========================================================================================
class ResNetSimCLR_ECOC(nn.Module):
    def __init__(self, base_model, out_dim, code_dim):
        super(ResNetSimCLR_ECOC, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(weights=None, num_classes=out_dim),
            "resnet50": models.resnet50(weights=None, num_classes=out_dim),
        }
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
        )
        self.codebook = nn.Linear(out_dim, code_dim, bias=False)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise Exception(f"Invalid backbone model name '{model_name}'.")
        return model

    def forward(self, x):
        h = self.backbone(x)
        c = self.codebook(h)
        c = torch.tanh(c)
        return h, c


# ==========================================================================================
# Main Experiment Logic adapted from run.py
# ==========================================================================================
def main():
    parser = argparse.ArgumentParser(description="SimCLR and ACL Pre-training")
    parser.add_argument(
        "--data", metavar="DIR", default="./datasets", help="path to dataset"
    )
    parser.add_argument(
        "--dataset-name", default="cifar10", help="dataset name", choices=["cifar10"]
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="model architecture (default: resnet18)",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.075,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training."
    )
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    parser.add_argument(
        "--fp16-precision",
        action="store_true",
        help="Whether or not to use 16-bit precision training.",
    )
    parser.add_argument(
        "--out_dim", default=128, type=int, help="feature dimension (default: 128)"
    )
    parser.add_argument(
        "--log-every-n-steps", default=100, type=int, help="Log every n steps"
    )
    parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
        help="softmax temperature (default: 0.5)",
    )
    parser.add_argument(
        "--n-views",
        default=2,
        type=int,
        metavar="N",
        help="Number of views for contrastive learning.",
    )
    parser.add_argument(
        "--model_type",
        default="acl",
        type=str,
        help="pre-training model type",
        choices=["simclr", "acl"],
    )
    parser.add_argument(
        "--csl_lambda",
        default=0.1,
        type=float,
        help="weight for column separation loss (ACL only)",
    )
    parser.add_argument(
        "--code_dim", default=64, type=int, help="codeword length for ECOC (ACL only)"
    )
    parser.add_argument(
        "--save_dir",
        default="checkpoints",
        type=str,
        help="directory to save checkpoints",
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check for CUDA
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        logging.info("CUDA is not available. Training on CPU.")

    # Data loading
    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Model definition
    if args.model_type == "acl":
        logging.info(
            f"Creating ACL model with backbone: {args.arch}, code_dim: {args.code_dim}"
        )
        model = ResNetSimCLR_ECOC(
            base_model=args.arch, out_dim=args.out_dim, code_dim=args.code_dim
        ).to(args.device)
    else:  # simclr
        # Simplified for this example to reuse the same model structure without the codebook part
        # A full implementation would use a different model class.
        raise NotImplementedError(
            "SimCLR model type is not implemented in this minimal script."
        )

    # Optimizer and scheduler
    optimizer = LARS(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=0, last_epoch=-1
    )

    # Loss function (InfoNCE)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    def info_nce_loss(features):
        labels = torch.cat(
            [torch.arange(args.batch_size) for i in range(args.n_views)], dim=0
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(args.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

        logits = logits / args.temperature
        return logits, labels

    def column_separation_loss(codebook):
        # L_csl encourages columns of the codebook to be dissimilar.
        # We want to minimize the cosine similarity between different columns,
        # which is equivalent to maximizing the angle between them.
        codebook_norm = F.normalize(codebook, p=2, dim=0)
        cos_sim = torch.mm(codebook_norm.t(), codebook_norm)
        # We sum the off-diagonal elements of the cosine similarity matrix.
        loss_csl = (
            torch.ones_like(cos_sim) - torch.eye(cos_sim.shape[0], device=args.device)
        ) * cos_sim
        return loss_csl.mean()

    # Training loop
    for epoch in range(args.epochs):
        for i, (images, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        ):
            images = torch.cat(images, dim=0).to(args.device)

            h, c = model(images)
            logits, labels = info_nce_loss(h)
            loss_infonce = criterion(logits, labels)

            # ACL specific loss
            loss_csl = column_separation_loss(model.codebook.weight)
            loss = loss_infonce + args.csl_lambda * loss_csl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_every_n_steps == 0:
                logging.info(
                    f"Epoch: [{epoch+1}/{args.epochs}], Step: [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, InfoNCE: {loss_infonce.item():.4f}, CSL: {loss_csl.item():.4f}"
                )

        scheduler.step()

        # Save checkpoint
        checkpoint_name = f"acl_checkpoint_{epoch+1:04d}.pth.tar"
        torch.save(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.save_dir, checkpoint_name),
        )
        logging.info(
            f"Saved checkpoint to {os.path.join(args.save_dir, checkpoint_name)}"
        )


if __name__ == "__main__":
    main()
