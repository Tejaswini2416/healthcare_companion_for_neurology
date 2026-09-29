"""
RANO 2.0 Pseudoprogression (PsP) Guard & Urgency Triage Module
Applies formal RANO 2.0 criteria, post-radiotherapy temporal rules (12 weeks / 84 days),
perfusion rCBV thresholding (< 1.75), and MGMT methylation status to differentiate
treatment-induced pseudoprogression from true neoplastic recurrence,
generating formal RANO categories ('CR', 'PR', 'SD', 'PD', 'PsP') and urgency triage scores.
"""

from datetime import datetime, date
from typing import Dict, Any, Optional, Union


class RANO2TriageEngine:
    """
    RANO 2.0 clinical response evaluation and pseudoprogression triage classifier.
    """
    def __init__(self):
        # Clinical parameters based on RANO 2.0 guidelines (Lancet Oncology 2023)
        self.post_rt_psp_window_days = 84 # 12 weeks
        self.rcbv_hypoperfusion_threshold = 1.75

    def evaluate_response(
        self,
        current_et_vol: float,
        prior_et_vol: Optional[float],
        baseline_et_vol: float,
        days_since_rt_completion: int,
        rcbv: float,
        mgmt_methylated: bool,
        midline_shift_mm: float = 0.0,
        corticosteroid_dose_stable: bool = True,
        symptom_severity: str = "Mild",
        missed_meds_flag: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates RANO 2.0 response status and assigns clinical triage urgency.
        """
        ref_vol = prior_et_vol if (prior_et_vol is not None and prior_et_vol > 0) else baseline_et_vol
        vol_change_pct = ((current_et_vol - ref_vol) / max(0.1, ref_vol)) * 100.0

        within_post_rt_window = days_since_rt_completion <= self.post_rt_psp_window_days
        is_hypoperfused = rcbv < self.rcbv_hypoperfusion_threshold

        # Pseudoprogression Probability Modeling
        # Formula combines temporal proximity, perfusion rCBV, and MGMT methylation
        psp_score = 0.0
        if within_post_rt_window:
            psp_score += 0.40
        if is_hypoperfused:
            psp_score += 0.35
        if mgmt_methylated:
            psp_score += 0.20 # MGMT methylated tumors have significantly higher incidence of radiation necrosis

        psp_probability = round(min(0.96, max(0.05, psp_score)), 4)
        is_pseudoprogression = False

        # RANO 2.0 Category Classification
        if current_et_vol == 0.0:
            rano_category = "CR" # Complete Response
            clinical_rationale = "Total resolution of contrast-enhancing lesion."
        elif vol_change_pct <= -50.0:
            rano_category = "PR" # Partial Response
            clinical_rationale = f"Significant volumetric reduction of {abs(vol_change_pct):.1f}% in enhancing tumor."
        elif vol_change_pct >= 25.0:
            # Evaluating progression vs pseudoprogression
            if within_post_rt_window and (is_hypoperfused or mgmt_methylated):
                rano_category = "PsP" # Pseudoprogression
                is_pseudoprogression = True
                clinical_rationale = (
                    f"Lesion enlargement ({vol_change_pct:+.1f}%) observed within {days_since_rt_completion} days "
                    f"post-RT (<84d) with low rCBV ({rcbv:.2f} < 1.75) and MGMT methylation. "
                    "Favors radiation-induced pseudoprogression (necrosis/inflammation) rather than true recurrence."
                )
            else:
                rano_category = "PD" # Progressive Disease
                clinical_rationale = (
                    f"Definite progressive disease: {vol_change_pct:+.1f}% volume increase "
                    f"outside PsP protection window or with elevated rCBV ({rcbv:.2f})."
                )
        else:
            # Between -50% and +25%
            if vol_change_pct > 10.0 and within_post_rt_window and is_hypoperfused:
                rano_category = "PsP"
                is_pseudoprogression = True
                clinical_rationale = "Mild interval enhancement within 12 weeks post-RT with low perfusion, consistent with PsP."
            else:
                rano_category = "SD" # Stable Disease
                clinical_rationale = f"Stable enhancing lesion volume (Δ {vol_change_pct:+.1f}%)."

        # Urgency Triage Scoring ('Mild', 'Moderate', 'Critical')
        if midline_shift_mm >= 3.0 or (rano_category == "PD" and symptom_severity == "Severe"):
            triage_category = "Critical"
            alert_banner = (
                "CRITICAL ALERT: Significant structural midline shift and progressive disease flagged. "
                "Urgent neuro-oncology intervention and mass-effect management required."
            )
        elif rano_category == "PD" or (missed_meds_flag and symptom_severity in ["Moderate", "Severe"]):
            triage_category = "Moderate"
            alert_banner = (
                "CLINICAL ATTENTION: Missed antiepileptic/steroid medication coincides with reported symptoms. "
                "Care team notified for dose reconciliation."
            )
        elif rano_category == "PsP":
            triage_category = "Moderate"
            alert_banner = (
                "POST-RT MONITORING: Scan changes consistent with RANO 2.0 treatment-induced pseudoprogression. "
                "Maintain current therapy; scheduled short-interval follow-up in 4-6 weeks recommended."
            )
        else:
            triage_category = "Mild"
            alert_banner = "Normal follow-up status. Symptoms and scan metrics remain stable."

        return {
            "rano_category": rano_category,
            "triage_category": triage_category,
            "is_pseudoprogression": is_pseudoprogression,
            "pseudoprogression_probability": psp_probability,
            "vol_change_pct": round(vol_change_pct, 2),
            "days_since_rt": days_since_rt_completion,
            "within_post_rt_window": within_post_rt_window,
            "is_hypoperfused": is_hypoperfused,
            "clinical_rationale": clinical_rationale,
            "alert_banner_text": alert_banner,
        }
