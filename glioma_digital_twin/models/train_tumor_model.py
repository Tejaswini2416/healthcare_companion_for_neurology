"""
Brain Tumor MRI Classification: Training, Testing, and Evaluation Pipeline
Trained on MRI scans from data/sample_scans (98 'no', 155 'yes')
Evaluates on unseen test set with comprehensive metrics, ROC-AUC, Confusion Matrix,
and generates explainable Grad-CAM heatmaps for the Glioma Digital Twin.
"""

import os
import sys
import io
import json
import time
import random
from typing import Dict, Any, Tuple, List

# Ensure safe UTF-8 output on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BrainMRIDataset(Dataset):
    """PyTorch Dataset for Brain Tumor MRI Scans."""
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Load image and ensure 3-channel RGB
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)

        return img_tensor, torch.tensor(label, dtype=torch.long), path


class BrainTumorCNN(nn.Module):
    """
    Brain Tumor Deep Convolutional Neural Network with ResNet Backbone
    and Attention-Guided Multi-Stage Classification Head.
    """
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            num_ftrs = base_model.fc.in_features
            # Remove original fc
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = num_ftrs
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_model = models.resnet34(weights=weights)
            num_ftrs = base_model.fc.in_features
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = num_ftrs
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classification Head with Layer Normalization & Dropout for Regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_maps = self.encoder(x)
        pooled = self.gap(feat_maps).view(x.size(0), -1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits

    def get_last_conv_layer(self):
        """Returns the final convolutional block for Grad-CAM activation mapping."""
        return self.encoder[-1]


def compute_gradcam(model: BrainTumorCNN, img_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
    """
    Generates Class Activation Heatmap (Grad-CAM) for visual explainability.
    Shows where the model localizes brain tumor pathology.
    """
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    target_layer = model.get_last_conv_layer()
    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_backward_hook(backward_hook)

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # Forward
    model.zero_grad()
    logits = model(img_tensor)
    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()

    # Backward
    score = logits[0, target_class]
    score.backward()

    # Clean hooks
    h_fwd.remove()
    h_bwd.remove()

    if not gradients or not activations:
        return np.zeros((224, 224), dtype=np.float32)

    grads = gradients[0][0].cpu().numpy() # [C, H, W]
    acts = activations[0][0].cpu().numpy() # [C, H, W]

    # Global average pooling of gradients as channel weights
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    # Resize CAM to (224, 224)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
    return np.array(cam_img) / 255.0


def scan_dataset_files(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Scans data directory for 'no' (label 0) and 'yes' (label 1) brain tumor scans.
    """
    # Priority search locations
    candidates = [
        os.path.join(data_dir, "sample_scans", "brain_tumor_dataset"),
        os.path.join(data_dir, "sample_scans"),
        data_dir,
    ]

    target_base = None
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "no")) and os.path.exists(os.path.join(cand, "yes")):
            target_base = cand
            break

    if target_base is None:
        raise FileNotFoundError(f"Could not locate 'no' and 'yes' tumor scan directories in {data_dir}")

    file_paths = []
    labels = []
    valid_exts = {".jpg", ".jpeg", ".png"}

    # Class 0: No Tumor
    no_dir = os.path.join(target_base, "no")
    for f in sorted(os.listdir(no_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            file_paths.append(os.path.join(no_dir, f))
            labels.append(0)

    # Class 1: Tumor Present
    yes_dir = os.path.join(target_base, "yes")
    for f in sorted(os.listdir(yes_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            file_paths.append(os.path.join(yes_dir, f))
            labels.append(1)

    print(f"[Dataset] Found {len(file_paths)} total MRI scans from: {target_base}")
    print(f"          - Class 0 (No Tumor):     {labels.count(0)}")
    print(f"          - Class 1 (Tumor Present): {labels.count(1)}")

    return file_paths, labels


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns augmentation transforms for training and standard transforms for validation/test.
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_test_transform


def train_and_evaluate(
    data_dir: str = "data",
    output_dir: str = "glioma_digital_twin/models/saved_models",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Full training, validation, testing, and benchmark execution.
    """
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    print(f"\n[Environment] PyTorch Device: {device} | Output Directory: {output_dir}")

    # 1. Dataset Discovery & Splitting
    file_paths, labels = scan_dataset_files(data_dir)

    # Stratified Train (70%), Val (15%), Test (15%) split
    train_paths, temp_paths, train_y, temp_y = train_test_split(
        file_paths, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_y, test_y = train_test_split(
        temp_paths, temp_y, test_size=0.50, stratify=temp_y, random_state=42
    )

    print(f"[Split] Train: {len(train_paths)} scans (No: {train_y.count(0)}, Yes: {train_y.count(1)})")
    print(f"        Val:   {len(val_paths)} scans (No: {val_y.count(0)}, Yes: {val_y.count(1)})")
    print(f"        Test:  {len(test_paths)} scans (No: {test_y.count(0)}, Yes: {test_y.count(1)})")

    train_tf, val_test_tf = get_transforms()

    train_dataset = BrainMRIDataset(train_paths, train_y, transform=train_tf)
    val_dataset = BrainMRIDataset(val_paths, val_y, transform=val_test_tf)
    test_dataset = BrainMRIDataset(test_paths, test_y, transform=val_test_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model Initialization & Class Weighting
    model = BrainTumorCNN(backbone="resnet18", pretrained=True, num_classes=2, dropout=0.35)
    model.to(device)

    # Class weights for CrossEntropyLoss to address class distribution
    class_counts = [train_y.count(0), train_y.count(1)]
    total_train = len(train_y)
    class_weights = torch.tensor([total_train / (2.0 * c) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 3. Training Loop
    print("\n" + "=" * 70)
    print("                STARTING MODEL TRAINING & VALIDATION                 ")
    print("=" * 70)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": []
    }

    best_val_auc = 0.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(lbls.cpu().numpy())

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = accuracy_score(train_targets, train_preds)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)

                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(lbls.cpu().numpy())
                val_probs.extend(probs)

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = accuracy_score(val_targets, val_preds)
        epoch_val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        epoch_val_auc = roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else 0.5

        scheduler.step(epoch_val_auc)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["val_f1"].append(round(epoch_val_f1, 4))
        history["val_auc"].append(round(epoch_val_auc, 4))

        is_best = epoch_val_auc > best_val_auc
        if is_best or best_model_state is None:
            best_val_auc = epoch_val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " * (Best Val AUC)"
        else:
            mark = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}% | "
              f"Val AUC: {epoch_val_auc:.4f}{mark}", flush=True)

    training_duration = round(time.time() - start_time, 2)
    print(f"\n[Training Complete] Elapsed Time: {training_duration}s | Peak Val AUC: {best_val_auc:.4f}")

    # Load best checkpoint
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    # 4. Rigorous Evaluation on UNSEEN Test Set
    print("\n" + "=" * 70)
    print("               EVALUATION ON UNSEEN TEST SET (38 SCANS)              ")
    print("=" * 70)

    test_preds = []
    test_targets = []
    test_probs = []
    test_image_paths = []
    test_embeddings = []

    with torch.no_grad():
        for imgs, lbls, paths in test_loader:
            imgs = imgs.to(device)
            feat = model.extract_features(imgs)
            logits = model.classifier(feat)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            test_preds.extend(preds.tolist())
            test_targets.extend(lbls.tolist())
            test_probs.extend(probs.tolist())
            test_image_paths.extend(paths)
            test_embeddings.extend(feat.cpu().numpy().tolist())

    # Calculate Complete Test Metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds, pos_label=1, zero_division=0)
    test_recall = recall_score(test_targets, test_preds, pos_label=1, zero_division=0) # Sensitivity
    test_f1 = f1_score(test_targets, test_preds, pos_label=1, zero_division=0)
    test_f1_macro = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    test_auc = roc_auc_score(test_targets, test_probs)
    test_pr_auc = average_precision_score(test_targets, test_probs)

    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = cm.ravel()
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    target_names = ["No Tumor (0)", "Tumor Present (1)"]
    clf_report = classification_report(test_targets, test_preds, target_names=target_names, output_dict=True)

    print(f"\n[Test Set Results Breakdown]")
    print(f"  • Overall Accuracy:         {test_acc * 100:.2f}%")
    print(f"  • Sensitivity / Recall:    {test_recall * 100:.2f}% (Tumor Detection Rate)")
    print(f"  • Specificity:             {test_specificity * 100:.2f}% (Healthy Scans Correctly Identified)")
    print(f"  • Precision (PPV):         {test_precision * 100:.2f}%")
    print(f"  • F1-Score (Tumor Class):  {test_f1 * 100:.2f}%")
    print(f"  • Macro F1-Score:          {test_f1_macro * 100:.2f}%")
    print(f"  • ROC-AUC Score:           {test_auc:.4f}")
    print(f"  • PR-AUC (Avg Precision):  {test_pr_auc:.4f}")
    print(f"\n[Confusion Matrix]")
    print(f"                Predicted No  |  Predicted Yes")
    print(f"  Actual No  :       {tn:2d}       |       {fp:2d}      (Total: {tn+fp})")
    print(f"  Actual Yes :       {fn:2d}       |       {tp:2d}      (Total: {fn+tp})")

    # 5. Comparative Machine Learning Benchmark (Random Forest & SVM on extracted deep features)
    print("\n[Benchmark] Training Classical ML Classifiers on Deep Bottleneck Features...")
    train_embeddings = []
    with torch.no_grad():
        for imgs, _, _ in DataLoader(train_dataset, batch_size=batch_size, shuffle=False):
            imgs = imgs.to(device)
            feat = model.extract_features(imgs).cpu().numpy()
            train_embeddings.extend(feat)

    train_X = np.array(train_embeddings)
    train_Y = np.array(train_y)
    test_X = np.array(test_embeddings)
    test_Y = np.array(test_targets)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(train_X, train_Y)
    rf_preds = rf.predict(test_X)
    rf_probs = rf.predict_proba(test_X)[:, 1]
    rf_acc = accuracy_score(test_Y, rf_preds)
    rf_f1 = f1_score(test_Y, rf_preds, average='macro')
    rf_auc = roc_auc_score(test_Y, rf_probs)

    # Support Vector Machine (RBF kernel)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(train_X, train_Y)
    svm_preds = svm.predict(test_X)
    svm_probs = svm.predict_proba(test_X)[:, 1]
    svm_acc = accuracy_score(test_Y, svm_preds)
    svm_f1 = f1_score(test_Y, svm_preds, average='macro')
    svm_auc = roc_auc_score(test_Y, svm_probs)

    benchmark_comparison = {
        "Deep_CNN_ResNet18": {
            "model_type": "Deep Convolutional Neural Network (ResNet18)",
            "test_accuracy": round(float(test_acc), 4),
            "test_precision": round(float(test_precision), 4),
            "test_recall": round(float(test_recall), 4),
            "test_specificity": round(float(test_specificity), 4),
            "test_f1": round(float(test_f1_macro), 4),
            "test_auc": round(float(test_auc), 4),
        },
        "Random_Forest": {
            "model_type": "Random Forest (100 Trees on 512D Embeddings)",
            "test_accuracy": round(float(rf_acc), 4),
            "test_precision": round(float(precision_score(test_Y, rf_preds, zero_division=0)), 4),
            "test_recall": round(float(recall_score(test_Y, rf_preds, zero_division=0)), 4),
            "test_specificity": round(float(confusion_matrix(test_Y, rf_preds)[0,0] / np.sum(confusion_matrix(test_Y, rf_preds)[0])), 4),
            "test_f1": round(float(rf_f1), 4),
            "test_auc": round(float(rf_auc), 4),
        },
        "Support_Vector_Machine": {
            "model_type": "Support Vector Machine (RBF Kernel)",
            "test_accuracy": round(float(svm_acc), 4),
            "test_precision": round(float(precision_score(test_Y, svm_preds, zero_division=0)), 4),
            "test_recall": round(float(recall_score(test_Y, svm_preds, zero_division=0)), 4),
            "test_specificity": round(float(confusion_matrix(test_Y, svm_preds)[0,0] / np.sum(confusion_matrix(test_Y, svm_preds)[0])), 4),
            "test_f1": round(float(svm_f1), 4),
            "test_auc": round(float(svm_auc), 4),
        }
    }

    # 6. Generate and Save Visualizations & Artifacts
    print("\n[Artifacts] Generating high-resolution diagnostic plots...")

    # Plot 1: Training & Validation Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, epochs + 1)
    axes[0].plot(epochs_range, history["train_loss"], 'o-', color='#2563eb', label='Train Loss', lw=2)
    axes[0].plot(epochs_range, history["val_loss"], 's--', color='#ef4444', label='Val Loss', lw=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].plot(epochs_range, [x * 100 for x in history["train_acc"]], 'o-', color='#10b981', label='Train Accuracy', lw=2)
    axes[1].plot(epochs_range, [x * 100 for x in history["val_acc"]], 's--', color='#8b5cf6', label='Val Accuracy', lw=2)
    axes[1].plot(epochs_range, [x * 100 for x in history["val_auc"]], '^-.', color='#f59e0b', label='Val ROC-AUC', lw=2)
    axes[1].set_title("Accuracy & AUC Performance vs. Epochs", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()

    # Plot 2: Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["No Tumor (0)", "Tumor Present (1)"],
            yticklabels=["No Tumor (0)", "Tumor Present (1)"],
            annot_kws={"size": 14, "weight": "bold"}
        )
    else:
        plt.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=14, weight='bold')
        plt.xticks([0, 1], ["No Tumor (0)", "Tumor Present (1)"])
        plt.yticks([0, 1], ["No Tumor (0)", "Tumor Present (1)"])
    plt.title(f"Test Set Confusion Matrix (Accuracy: {test_acc*100:.1f}%)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Diagnosis", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Diagnosis", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Plot 3: ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    axes[0].plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'Deep CNN (AUC = {test_auc:.4f})')
    # SVM & RF ROC
    svm_fpr, svm_tpr, _ = roc_curve(test_Y, svm_probs)
    rf_fpr, rf_tpr, _ = roc_curve(test_Y, rf_probs)
    axes[0].plot(svm_fpr, svm_tpr, color='#10b981', lw=1.8, linestyle='--', label=f'SVM (AUC = {svm_auc:.4f})')
    axes[0].plot(rf_fpr, rf_tpr, color='#f59e0b', lw=1.8, linestyle=':', label=f'Random Forest (AUC = {rf_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Baseline')
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=':', alpha=0.6)

    prec_pts, rec_pts, _ = precision_recall_curve(test_targets, test_probs)
    axes[1].plot(rec_pts, prec_pts, color='#8b5cf6', lw=2.5, label=f'Deep CNN (PR-AUC = {test_pr_auc:.4f})')
    axes[1].set_title("Precision-Recall Curve (PR-AUC)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (PPV)", fontsize=11)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_pr_path = os.path.join(output_dir, "roc_pr_curves.png")
    plt.savefig(roc_pr_path, dpi=200)
    plt.close()

    # Plot 4: Sample Test Predictions with Grad-CAM Explainability
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # Select 4 representative 'No Tumor' and 4 'Tumor' scans
    no_indices = [i for i, y in enumerate(test_targets) if y == 0][:4]
    yes_indices = [i for i, y in enumerate(test_targets) if y == 1][:4]
    sample_indices = no_indices + yes_indices

    for i, idx in enumerate(sample_indices[:8]):
        r, c = i // 4, i % 4
        p_path = test_image_paths[idx]
        true_label = test_targets[idx]
        pred_label = test_preds[idx]
        prob = test_probs[idx]

        with Image.open(p_path) as raw_img:
            raw_rgb = raw_img.convert("RGB").resize((224, 224))
            raw_np = np.array(raw_rgb)

        img_tensor = val_test_tf(raw_rgb)
        cam = compute_gradcam(model, img_tensor, target_class=pred_label)

        # Overlay heatmap
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = (0.6 * (raw_np / 255.0) + 0.4 * heatmap)
        overlay = np.clip(overlay, 0.0, 1.0)

        axes[r, c].imshow(overlay)
        status_sym = "[CORRECT]" if true_label == pred_label else "[MISSED]"
        true_txt = "Tumor" if true_label == 1 else "No Tumor"
        pred_txt = "Tumor" if pred_label == 1 else "No Tumor"
        conf = prob if pred_label == 1 else (1.0 - prob)

        axes[r, c].set_title(
            f"{status_sym} True: {true_txt}\nPred: {pred_txt} ({conf*100:.1f}%)",
            fontsize=10,
            fontweight='bold',
            color='darkgreen' if true_label == pred_label else 'darkred'
        )
        axes[r, c].axis('off')

    plt.suptitle("Sample Test Set Predictions & Grad-CAM Anatomical Saliency Heatmaps", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    gradcam_grid_path = os.path.join(output_dir, "test_predictions_gradcam.png")
    plt.savefig(gradcam_grid_path, dpi=200)
    plt.close()

    # 7. Compile Final Metric Payload & Save Weights
    model_save_path = os.path.join(output_dir, "best_tumor_classifier.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "backbone": "resnet18",
        "num_classes": 2,
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_precision), 4),
            "recall_sensitivity": round(float(test_recall), 4),
            "specificity": round(float(test_specificity), 4),
            "f1_score": round(float(test_f1), 4),
            "macro_f1": round(float(test_f1_macro), 4),
            "roc_auc": round(float(test_auc), 4),
            "pr_auc": round(float(test_pr_auc), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        },
        "training_history": history,
        "seed": 42
    }, model_save_path)
    print(f"\n[Checkpoint Saved] Full model and metadata saved to: {model_save_path}")

    # Build Sample Test Results for Inspection
    test_sample_results = []
    for idx in range(len(test_targets)):
        pred_l = test_preds[idx]
        true_l = test_targets[idx]
        p_val = test_probs[idx]
        conf_val = p_val if pred_l == 1 else (1.0 - p_val)
        test_sample_results.append({
            "filename": os.path.basename(test_image_paths[idx]),
            "ground_truth": "Tumor Present (1)" if true_l == 1 else "No Tumor (0)",
            "predicted_label": "Tumor Present (1)" if pred_l == 1 else "No Tumor (0)",
            "tumor_probability": round(float(p_val), 4),
            "confidence": round(float(conf_val), 4),
            "correct": bool(pred_l == true_l)
        })

    results_payload = {
        "dataset_summary": {
            "total_images": len(file_paths),
            "class_0_no_tumor": labels.count(0),
            "class_1_tumor_present": labels.count(1),
            "train_set_size": len(train_paths),
            "validation_set_size": len(val_paths),
            "test_set_size": len(test_paths),
        },
        "test_evaluation_metrics": {
            "accuracy": round(float(test_acc), 4),
            "accuracy_pct": round(float(test_acc * 100), 2),
            "precision": round(float(test_precision), 4),
            "precision_pct": round(float(test_precision * 100), 2),
            "recall_sensitivity": round(float(test_recall), 4),
            "recall_sensitivity_pct": round(float(test_recall * 100), 2),
            "specificity": round(float(test_specificity), 4),
            "specificity_pct": round(float(test_specificity * 100), 2),
            "f1_score": round(float(test_f1), 4),
            "f1_score_pct": round(float(test_f1 * 100), 2),
            "macro_f1": round(float(test_f1_macro), 4),
            "roc_auc": round(float(test_auc), 4),
            "pr_auc": round(float(test_pr_auc), 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "total_test_samples": int(tn + fp + fn + tp)
            },
            "classification_report": clf_report
        },
        "benchmark_comparison": benchmark_comparison,
        "training_metadata": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "device": str(device),
            "training_duration_seconds": training_duration,
            "history": history
        },
        "artifact_paths": {
            "model_checkpoint": model_save_path,
            "training_curves_plot": curves_path,
            "confusion_matrix_plot": cm_path,
            "roc_pr_curves_plot": roc_pr_path,
            "gradcam_grid_plot": gradcam_grid_path
        },
        "test_sample_predictions": test_sample_results
    }

    metrics_json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(metrics_json_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"[Results JSON Saved] Complete evaluation results saved to: {metrics_json_path}")
    return results_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and Evaluate Brain Tumor Classifier on Dataset")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data folder")
    parser.add_argument("--output-dir", type=str, default="glioma_digital_twin/models/saved_models", help="Output path")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_and_evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
