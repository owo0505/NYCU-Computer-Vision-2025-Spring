import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class MyResNeSt(nn.Module):
    def __init__(self, model_name="resnest50d",
                 num_classes=100, pretrained=True):

        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        in_features = self.backbone.num_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Generate MixUp data:
    x, y: Original batch of images and labels
    alpha: Parameter for the Beta distribution
    Returns (mixed_x, y_a, y_b, lam)
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    Generate CutMix data:
    x, y: Original batch of images and labels
    alpha: Parameter for the Beta distribution
    Returns (cutmix_x, y_a, y_b, lam)
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(device)

    # Randomly generate the patch region
    cut_rat = np.sqrt(lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # Randomly choose the location to paste
    cy = np.random.randint(0, H)
    cx = np.random.randint(0, W)

    # Calculate the actual region (upper and lower boundaries)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    # Paste the patch from the image at the generated index into x
    x_clone = x.clone()
    x_clone[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lam (because the actual patch may be truncated)
    area = (y2 - y1) * (x2 - x1)
    lam = 1 - area / (H * W)

    y_a, y_b = y, y[index]
    return x_clone, y_a, y_b, lam


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    Combine two labels y_a and y_b with weight lam using the given criterion.
    You can also use cross-entropy with label smoothing here.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def generate_confusion_matrix_with_heatmap(model, val_loader, device):
    """
    Generates a confusion matrix heatmap for the given model and data loader.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Create a figure with a larger size
    plt.figure(figsize=(12, 12))

    # Plot the heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=range(100), yticklabels=range(100))
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save and show
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


def main():
    data_dir = "/kaggle/input/dl-hw1-data/data"
    num_classes = 100
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-4
    alpha = 1.0  # Beta distribution parameter for MixUp / CutMix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Transforms: include RandAugment
    train_transform = transforms.Compose([
        RandAugment(num_ops=2, magnitude=7),  # RandAugment
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(os.path.join(data_dir, "train"),
                                transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, "val"),
                              transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # 2) Create MyResNeSt model
    model_name = "resnest200e"
    model = MyResNeSt(model_name=model_name,
                      num_classes=num_classes, pretrained=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Using {model_name}, Total Params: {total_params/1e6:.2f} M")

    # 3) Define Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=num_epochs,
                                                     eta_min=1e-6)
    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # ---- Training ----
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Use either MixUp or CutMix based on a random probability
            p = random.random()
            if p < 0.5:
                # MixUp
                mixed_x, y_a, y_b, lam = mixup_data(inputs, labels,
                                                    alpha=alpha, device=device)
                outputs = model(mixed_x)
                loss = mixup_cutmix_criterion(criterion, outputs,
                                              y_a, y_b, lam)
            else:
                # CutMix
                cutmix_x, y_a, y_b, lam = cutmix_data(inputs, labels,
                                                      alpha=alpha,
                                                      device=device)
                outputs = model(cutmix_x)
                loss = mixup_cutmix_criterion(criterion,
                                              outputs, y_a, y_b, lam)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss_epoch = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss_epoch += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)

        val_loss_epoch = val_loss_epoch / val_total
        val_acc_epoch = val_corrects / val_total
        print(f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}")

        # Store metrics for plotting
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)

        scheduler.step()

        # ---- Save the best model ----
        if val_acc_epoch > best_acc:
            best_acc = val_acc_epoch
            torch.save(model.state_dict(), f"best_epoch_{epoch+1}_resnest.pth")
            print(f"  Saved best model with val_acc={val_acc_epoch:.4f}")

        print("-----------")

    print(f"Best Val Acc: {best_acc:.4f}")

    # ---- Plot Training Curves ----
    epochs_range = list(range(1, num_epochs + 1))

    # Plot Loss Curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()

    # Plot Accuracy Curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accs, label="Training Accuracy", marker='o')
    plt.plot(epochs_range, val_accs, label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.show()

    # ---- Generate Confusion Matrix ----
    # We use the final (current) model for confusion matrix visualization.
    # If you prefer to use the best model, reload it from the saved file.
    generate_confusion_matrix_with_heatmap(model, val_loader, device)


if __name__ == "__main__":
    main()
