"""
Inference & Deployment Wrapper for Trained Brain Tumor MRI Classifier
Provides real-time inference, risk stratification, Grad-CAM explainability,
and integration with the Multimodal Glioma Digital Twin.
"""

import os
import json
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from .train_tumor_model import BrainTumorCNN, compute_gradcam


class BrainTumorClassifier:
    """
    Production-ready Brain Tumor Classifier.
    Loads trained weights from data/sample_scans training run and provides:
    - Binary classification: 'Tumor Detected' (1) vs 'No Tumor' (0)
    - Tumor probability and diagnostic confidence
    - Grad-CAM saliency heatmaps
    - 512D deep feature embedding for multimodal digital twin fusion
    """
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, "saved_models", "best_tumor_classifier.pt")
        self.model_path = model_path or default_model_path
        self.metrics_path = os.path.join(os.path.dirname(self.model_path), "evaluation_results.json")

        self.model = BrainTumorCNN(backbone="resnet18", pretrained=False, num_classes=2, dropout=0.35)
        self.is_trained = False

        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.is_trained = True
                print(f"[BrainTumorClassifier] Successfully loaded trained weights from: {self.model_path}")
            except Exception as e:
                print(f"[BrainTumorClassifier] Warning: Could not load weights ({e}). Initializing base model.")
        else:
            print(f"[BrainTumorClassifier] Warning: Model weights not found at {self.model_path}. Please run training first.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        generate_gradcam: bool = True
    ) -> Dict[str, Any]:
        """
        Classifies an MRI scan and returns tumor probability, diagnosis, and explainability.
        """
        # Convert input to PIL RGB
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
            filename = os.path.basename(image_input)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                pil_img = Image.fromarray((image_input * 255).astype(np.uint8) if image_input.max() <= 1.0 else image_input.astype(np.uint8)).convert("RGB")
            else:
                raise ValueError("Unsupported ndarray shape for image")
            filename = "uploaded_scan.png"
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            filename = "image_input"
        else:
            raise TypeError("Unsupported image input type")

        raw_rgb = pil_img.resize((224, 224))
        tensor_img = self.transform(raw_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.extract_features(tensor_img)
            logits = self.model.classifier(feat)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        p_no = float(probs[0])
        p_yes = float(probs[1])
        pred_class = 1 if p_yes >= 0.50 else 0
        confidence = p_yes if pred_class == 1 else p_no

        if pred_class == 1:
            diagnosis = "Brain Tumor / Neoplasm Detected"
            risk_category = "High Risk" if p_yes > 0.85 else "Moderate Risk"
            alert_color = "red"
        else:
            diagnosis = "No Tumor Detected (Healthy / Normal Brain MRI)"
            risk_category = "Low Risk"
            alert_color = "green"

        gradcam_map = None
        if generate_gradcam:
            try:
                gradcam_map = compute_gradcam(self.model, tensor_img.squeeze(0), target_class=pred_class)
            except Exception as e:
                print(f"[Grad-CAM error]: {e}")
                gradcam_map = None

        return {
            "filename": filename,
            "predicted_class": pred_class,
            "diagnosis": diagnosis,
            "tumor_probability": round(p_yes, 4),
            "tumor_probability_pct": round(p_yes * 100.0, 1),
            "no_tumor_probability": round(p_no, 4),
            "no_tumor_probability_pct": round(p_no * 100.0, 1),
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "risk_category": risk_category,
            "alert_color": alert_color,
            "raw_image_pil": raw_rgb,
            "gradcam_heatmap": gradcam_map,
            "deep_feature_embedding": feat.squeeze(0).cpu().numpy().tolist()
        }

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Loads benchmark metrics and test evaluation results."""
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r") as f:
                return json.load(f)
        return {"error": "Evaluation metrics not yet generated. Please train model."}
