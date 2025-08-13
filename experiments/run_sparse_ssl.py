import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import medmnist
from medmnist import INFO
from torchvision.models import resnet18, ResNet18_Weights

# --- Model Architectures (Adapted from SPARSE Paper Description) ---


class Generator(nn.Module):
    """A standard U-Net generator, as described in the SPARSE paper."""

    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        def down_block(in_f, out_f, normalize=True, bn=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1)]
            if normalize:
                if bn:
                    layers.append(nn.BatchNorm2d(out_f, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def up_block(in_f, out_f, bn=True):
            layers = [nn.ConvTranspose2d(in_f, out_f, 4, 2, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_f, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.down1 = nn.Sequential(*down_block(in_channels, 64, normalize=False))
        self.down2 = nn.Sequential(*down_block(64, 128))
        self.down3 = nn.Sequential(*down_block(128, 256))
        self.down4 = nn.Sequential(*down_block(256, 512))
        self.down5 = nn.Sequential(*down_block(512, 512))

        self.up1 = nn.Sequential(*up_block(512, 512))
        self.up2 = nn.Sequential(*up_block(1024, 256))
        self.up3 = nn.Sequential(*up_block(512, 128))
        self.up4 = nn.Sequential(*up_block(256, 64))

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], 1))
        u3 = self.up3(torch.cat([u2, d3], 1))
        u4 = self.up4(torch.cat([u3, d2], 1))

        return self.final_up(torch.cat([u4, d1], 1))


class Discriminator(nn.Module):
    """A PatchGAN discriminator with a dual head for authenticity and classification, as described in SPARSE."""

    def __init__(self, in_channels=3, num_classes=8):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        # Output for authenticity (real/fake)
        self.adv_layer = nn.Sequential(nn.Conv2d(512, 1, 1, padding=0))
        # Output for classification
        self.cls_layer = nn.Sequential(
            nn.Conv2d(512, num_classes, 1, padding=0), nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, img):
        out = self.model(img)
        validity = self.adv_layer(out)
        label = self.cls_layer(out)
        label = label.view(label.size(0), -1)
        return validity, label


class Classifier(nn.Module):
    """A simple ResNet18 classifier, replacing EfficientNet-B3 for minimality."""

    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


# --- Data Preparation (Inspired by SOURCE_DOCKERFILE) ---


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Assumes root_dir contains class subdirectories
        for i, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_dataset(base_path, data_name, n_shots):
    print(f"Preparing dataset: {data_name} with {n_shots} shots")
    data_info = INFO[data_name]
    num_classes = len(data_info["label"])
    DataClass = getattr(medmnist, data_info["python_class"])

    # Download data
    train_dataset = DataClass(split="train", download=True, as_rgb=True)

    # Setup paths
    sup_path = os.path.join(base_path, "supervised")
    unsup_path = os.path.join(base_path, "unsupervised")
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(sup_path)
    os.makedirs(unsup_path)

    # Create class directories
    for i in range(num_classes):
        os.makedirs(os.path.join(sup_path, str(i)))
        os.makedirs(os.path.join(unsup_path, str(i)))

    # Separate images by class
    images_by_class = [[] for _ in range(num_classes)]
    for img, label in train_dataset:
        images_by_class[label[0]].append(img)

    # Create few-shot supervised and unsupervised sets
    for i in range(num_classes):
        random.shuffle(images_by_class[i])
        sup_imgs = images_by_class[i][:n_shots]
        unsup_imgs = images_by_class[i][n_shots:]

        for j, img in enumerate(sup_imgs):
            img.save(os.path.join(sup_path, str(i), f"img_{j}.png"))

        for j, img in enumerate(unsup_imgs):
            img.save(os.path.join(unsup_path, str(i), f"img_{j}.png"))

    print(
        f"Dataset ready. Supervised: {n_shots*num_classes} images. Unsupervised: {len(train_dataset) - n_shots*num_classes} images."
    )
    return num_classes


# --- Main Training Script ---


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare Data
    data_path = os.path.join(args.data_root, args.data_name)
    num_classes = prepare_dataset(data_path, args.data_name, args.shots)

    # Models
    generator = Generator().to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)
    classifier = Classifier(num_classes=num_classes).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_C = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Losses
    adversarial_loss = nn.MSELoss().to(device)
    classification_loss = nn.CrossEntropyLoss().to(device)

    # DataLoaders
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    sup_dataset = CustomImageDataset(
        os.path.join(data_path, "supervised"), transform=transform
    )
    unsup_dataset = CustomImageDataset(
        os.path.join(data_path, "unsupervised"), transform=transform
    )
    sup_loader = DataLoader(sup_dataset, batch_size=args.batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=args.batch_size, shuffle=True)

    # Training Loop
    for epoch in range(args.epochs):
        # --- Supervised Phase ---
        for i, (sup_imgs, sup_labels) in enumerate(
            tqdm(sup_loader, desc=f"Epoch {epoch+1}/{args.epochs} [S]")
        ):
            sup_imgs, sup_labels = sup_imgs.to(device), sup_labels.to(device)

            # Train Classifier
            optimizer_C.zero_grad()
            c_preds = classifier(sup_imgs)
            c_loss = classification_loss(c_preds, sup_labels)
            c_loss.backward()
            optimizer_C.step()

            if i % 10 == 0:
                print(
                    f"[Epoch {epoch+1}] [Supervised] [Classifier Loss: {c_loss.item():.4f}]"
                )

        # --- Unsupervised Phase (every n epochs) ---
        if (epoch + 1) % args.n_unsupervised == 0:
            for i, (unsup_imgs, _) in enumerate(
                tqdm(unsup_loader, desc=f"Epoch {epoch+1}/{args.epochs} [U]")
            ):
                unsup_imgs = unsup_imgs.to(device)

                # Create pseudo-labels using the classifier
                with torch.no_grad():
                    c_preds = classifier(unsup_imgs)
                    d_validity, d_preds = discriminator(unsup_imgs)
                    # Ensemble prediction (simple average)
                    avg_preds = (
                        F.softmax(c_preds, dim=1) + F.softmax(d_preds, dim=1)
                    ) / 2
                    confidence, pseudo_labels = torch.max(avg_preds, 1)

                # Filter by confidence threshold
                confident_mask = (confidence > args.percentile).squeeze()
                if confident_mask.sum() == 0:
                    continue

                pseudo_labels = pseudo_labels[confident_mask]
                target_imgs = unsup_imgs[confident_mask]

                # --- Train Generator & Discriminator ---
                valid = torch.ones(
                    target_imgs.size(0), 1, 1, 1, device=device, requires_grad=False
                )
                fake = torch.zeros(
                    target_imgs.size(0), 1, 1, 1, device=device, requires_grad=False
                )

                # Generate images conditioned on pseudo-labels (simplified: we translate real images)
                # A true class-conditioned GAN would take noise + label. Here we do img-to-img translation.
                gen_imgs = generator(target_imgs)

                # Train Generator
                optimizer_G.zero_grad()
                g_adv_loss = adversarial_loss(discriminator(gen_imgs)[0], valid)
                g_loss = g_adv_loss
                g_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_validity, real_label = discriminator(target_imgs)
                fake_validity, _ = discriminator(gen_imgs.detach())

                d_real_adv_loss = adversarial_loss(real_validity, valid)
                d_fake_adv_loss = adversarial_loss(fake_validity, fake)
                d_adv_loss = (d_real_adv_loss + d_fake_adv_loss) / 2
                d_cls_loss = classification_loss(real_label, pseudo_labels)

                d_loss = (d_adv_loss + d_cls_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                if i % 50 == 0:
                    print(
                        f"[Epoch {epoch+1}] [Unsupervised] [D Loss: {d_loss.item():.4f}] [G Loss: {g_loss.item():.4f}]"
                    )

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPARSE SSL Training Experiment")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument(
        "--img_size", type=int, default=32, help="size of each image dimension"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="bloodmnist",
        help="name of the medmnist dataset",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="root directory for datasets"
    )
    parser.add_argument(
        "--shots", type=int, default=5, help="number of labeled samples per class"
    )
    parser.add_argument(
        "--n_unsupervised",
        type=int,
        default=5,
        help="run unsupervised training every N epochs",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.75,
        help="confidence threshold for pseudo-labeling",
    )
    args = parser.parse_args()

    main(args)
