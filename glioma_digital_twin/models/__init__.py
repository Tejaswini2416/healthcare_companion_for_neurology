"""
Glioma Digital Twin - Machine Learning & Biophysical Models Package
"""

# Neutralize PyTorch torch.classes inspection conflict with Streamlit local_sources_watcher
try:
    import torch
    if hasattr(torch, "_classes") and hasattr(torch._classes, "_Classes"):
        _orig_classes_getattr = torch._classes._Classes.__getattr__
        def _safe_classes_getattr(self, attr):
            if attr in ("__path__", "_path", "__file__", "__loader__") or attr.startswith("__"):
                raise AttributeError(f"'_Classes' object has no attribute '{attr}'")
            return _orig_classes_getattr(self, attr)
        torch._classes._Classes.__getattr__ = _safe_classes_getattr
except Exception:
    pass

from .mri_segmenter import SegResNetSegmenter, SyntheticBraTSGenerator
from .clinical_nlp import ClinicalNLPEngine
from .biophysical_solver import FisherKolmogorovSolver
from .triage_engine import RANO2TriageEngine
from .multimodal_twin import MultimodalTwinFusion
from .tumor_classifier import BrainTumorClassifier
from .train_tumor_model import BrainTumorCNN

__all__ = [
    "SegResNetSegmenter",
    "SyntheticBraTSGenerator",
    "ClinicalNLPEngine",
    "FisherKolmogorovSolver",
    "RANO2TriageEngine",
    "MultimodalTwinFusion",
    "BrainTumorClassifier",
    "BrainTumorCNN",
]
