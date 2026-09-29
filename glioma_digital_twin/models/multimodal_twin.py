"""
Multimodal Intermediate Attention Fusion Module (PyTorch nn.Module)
Fuses:
  - 128-dim SegResNet 3D MRI latent bottleneck vector
  - 11-dim ClinicalBERT narrative report embedding
  - 8-dim Structured EHR, symptoms, and medication adherence vector
Cross-attention transformer over modality tokens with LayerNorm, residual projection,
and multi-task prediction heads generating a unified 64-dimensional Patient Health Twin vector.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union


class MultimodalIntermediateAttentionFusion(nn.Module):
    """
    PyTorch nn.Module implementing intermediate cross-modal attention fusion
    across MRI imaging, Clinical NLP, and Structured EHR / symptom signals.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Modality projection layers mapping each input to common d_model space (64-dim)
        self.proj_mri = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.proj_nlp = nn.Sequential(
            nn.Linear(11, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.proj_ehr = nn.Sequential(
            nn.Linear(8, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Modality type embeddings (learned token identifiers)
        self.modality_embed = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        # Multi-Head Cross-Modal Attention Layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward subnetwork with residual connection
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Attention pooling query to aggregate modality tokens into unified twin
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=True)

        # Final twin projection & Multi-Task Diagnostic Heads
        self.twin_proj = nn.Linear(d_model, d_model)
        self.progression_risk_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.triage_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Mild, Moderate, Critical
        )

    def forward(
        self,
        mri_latent: torch.Tensor,   # [B, 128]
        nlp_embed: torch.Tensor,    # [B, 11]
        ehr_features: torch.Tensor  # [B, 8]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            - twin_vector: [B, 64] unified patient health twin representation
            - progression_risk: [B, 1] scalar score [0, 1]
            - triage_logits: [B, 3] class logits
            - modality_weights: [B, 3] attention weights for each modality
        """
        B = mri_latent.size(0)

        # 1. Project modalities to 64-dim tokens
        tok_mri = self.proj_mri(mri_latent).unsqueeze(1) # [B, 1, 64]
        tok_nlp = self.proj_nlp(nlp_embed).unsqueeze(1)   # [B, 1, 64]
        tok_ehr = self.proj_ehr(ehr_features).unsqueeze(1)# [B, 1, 64]

        # Stack into sequence of tokens: [B, 3, 64]
        tokens = torch.cat([tok_mri, tok_nlp, tok_ehr], dim=1) + self.modality_embed

        # 2. Cross-Modal Attention with residual connection & LayerNorm
        attn_out, attn_weights = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # 3. Position-wise Feed Forward with residual & LayerNorm
        tokens = self.norm2(tokens + self.ffn(tokens))

        # 4. Attention Pooling over the 3 modalities
        query = self.pool_query.repeat(B, 1, 1) # [B, 1, 64]
        pooled_out, pool_weights = self.pool_attn(query, tokens, tokens)
        modality_weights = pool_weights.squeeze(1) # [B, 3]

        # 5. Output unified 64-dim Twin Representation
        twin_vector = F.normalize(self.twin_proj(pooled_out.squeeze(1)), p=2, dim=1) # [B, 64]

        # 6. Multi-task predictions
        progression_risk = self.progression_risk_head(twin_vector)
        triage_logits = self.triage_head(twin_vector)

        return twin_vector, progression_risk, triage_logits, modality_weights


class MultimodalTwinFusion:
    """
    Inference orchestrator for the Patient Digital Twin Multimodal Fusion Layer.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = MultimodalIntermediateAttentionFusion(d_model=64, n_heads=4).to(self.device)
        self.model.eval()

    @staticmethod
    def construct_ehr_vector(
        age: int,
        baseline_kps: int,
        idh_status: str,
        mgmt_methylated: bool,
        adherence_pct: float,
        missed_meds_flag: bool,
        symptom_severity: str,
        has_seizure_or_aura: bool
    ) -> List[float]:
        """
        Encodes structured EHR, symptoms, and adherence into an 8-dimensional normalized feature vector.
        """
        # 0: Normalized Age [0-1]
        v0 = float(np.clip(age / 100.0, 0.0, 1.0))

        # 1: Normalized Baseline KPS [0-1]
        v1 = float(np.clip(baseline_kps / 100.0, 0.1, 1.0))

        # 2: IDH status (1.0 for Mutant, 0.0 for Wild-Type)
        v2 = 1.0 if idh_status.lower() in ["mutant", "idh-mutant"] else 0.0

        # 3: MGMT methylation (1.0 for Methylated, 0.0 for Unmethylated)
        v3 = 1.0 if mgmt_methylated else 0.0

        # 4: Medication adherence percentage [0-1]
        v4 = float(np.clip(adherence_pct / 100.0, 0.0, 1.0))

        # 5: Missed dose flag (1.0 if missed, 0.0 otherwise)
        v5 = 1.0 if missed_meds_flag else 0.0

        # 6: Symptom severity scale (0.0: none/mild, 0.5: moderate, 1.0: severe)
        severity_map = {"none": 0.0, "mild": 0.25, "moderate": 0.65, "severe": 1.0}
        v6 = severity_map.get(symptom_severity.lower(), 0.3)

        # 7: Acute neurologic aura / seizure indicator
        v7 = 1.0 if has_seizure_or_aura else 0.0

        return [round(x, 4) for x in [v0, v1, v2, v3, v4, v5, v6, v7]]

    def fuse(
        self,
        mri_latent_128: List[float],
        nlp_embed_11: List[float],
        ehr_vector_8: List[float]
    ) -> Dict[str, Any]:
        """
        Executes intermediate cross-attention fusion over the 3 modalities.
        Returns:
            - latent_twin_vector: 64-dimensional float vector
            - progression_risk_score: float [0, 1]
            - triage_category: 'Mild', 'Moderate', or 'Critical'
            - modality_influence: dict mapping modality -> relative percentage weight
        """
        t_mri = torch.tensor([mri_latent_128], dtype=torch.float32, device=self.device)
        t_nlp = torch.tensor([nlp_embed_11], dtype=torch.float32, device=self.device)
        t_ehr = torch.tensor([ehr_vector_8], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            twin_vec, prog_risk, triage_logits, mod_weights = self.model(t_mri, t_nlp, t_ehr)

            twin_list = [round(float(x), 5) for x in twin_vec.squeeze(0).cpu().numpy()]
            prog_risk_val = round(float(prog_risk.item()), 4)

            # Softmax on triage logits
            triage_probs = F.softmax(triage_logits, dim=-1).squeeze(0).cpu().numpy()
            triage_labels = ["Mild", "Moderate", "Critical"]
            triage_idx = int(np.argmax(triage_probs))
            triage_cat = triage_labels[triage_idx]

            # Modality weights (MRI, NLP, EHR)
            weights = mod_weights.squeeze(0).cpu().numpy()
            total_w = float(np.sum(weights))
            if total_w > 0:
                weights = weights / total_w
            else:
                weights = [0.333, 0.333, 0.334]

            mod_influence = {
                "mri_imaging": round(float(weights[0]) * 100.0, 1),
                "clinical_nlp": round(float(weights[1]) * 100.0, 1),
                "structured_ehr": round(float(weights[2]) * 100.0, 1),
            }

        return {
            "latent_twin_vector": twin_list, # 64-dim vector
            "progression_risk_score": prog_risk_val,
            "triage_category": triage_cat,
            "triage_probabilities": {
                "Mild": round(float(triage_probs[0]), 3),
                "Moderate": round(float(triage_probs[1]), 3),
                "Critical": round(float(triage_probs[2]), 3),
            },
            "modality_influence_pct": mod_influence,
        }
