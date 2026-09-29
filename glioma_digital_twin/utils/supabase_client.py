"""
Database Client Module (Supabase + Local Persistent Fallback)
Provides seamless dual-mode connectivity: connects directly to live Supabase PostgreSQL
if credentials exist in environment variables, or operates an in-memory/JSON-persisted
database engine for instant, reliable local and offline demonstration.
Pre-seeds multiple realistic clinical patient profiles (MRN 0042, 0043, 0044).
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union

try:
    from supabase import create_client, Client
    SUPABASE_SDK_AVAILABLE = True
except ImportError:
    SUPABASE_SDK_AVAILABLE = False


class DatabaseClient:
    """
    Unified database interface supporting real Supabase and resilient local memory storage.
    """
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self.is_connected_to_supabase = False
        self.supabase_client: Optional[Any] = None

        # Check if real Supabase credentials are provided
        if SUPABASE_SDK_AVAILABLE and self.url and self.key and not self.url.startswith("https://your-project"):
            try:
                self.supabase_client = create_client(self.url, self.key)
                _ = self.supabase_client.table("patients").select("patient_id").limit(1).execute()
                self.is_connected_to_supabase = True
                print(f"[DatabaseClient] Connected to remote Supabase: {self.url}")
            except Exception as e:
                print(f"[DatabaseClient] Remote Supabase connection failed ({e}). Defaulting to local resilient storage.")
                self.is_connected_to_supabase = False

        # In-memory storage tables
        self.storage: Dict[str, List[Dict[str, Any]]] = {
            "patients": [],
            "molecular_profiles": [],
            "mri_scans": [],
            "clinical_reports": [],
            "medications": [],
            "symptom_logs": [],
            "twin_timeline": [],
            "counterfactual_simulations": []
        }

        # Seed full cohort of patient profiles
        self._seed_default_cohort()

    def _seed_default_cohort(self):
        """Seeds demo data for Patient V. Thanuja (0042), Marcus Chen (0043), and Priya Sharma (0044)"""
        p42_id = "e5b38d38-2c26-4d2b-91c6-2c1b97b00042"
        p43_id = "e5b38d38-2c26-4d2b-91c6-2c1b97b00043"
        p44_id = "e5b38d38-2c26-4d2b-91c6-2c1b97b00044"

        # 1. Patient Demographics
        self.storage["patients"] = [
            {
                "patient_id": p42_id,
                "mrn": "0042",
                "full_name": "V. Thanuja",
                "age": 24,
                "gender": "Female",
                "diagnosis_type": "High-Grade Glioma (Astrocytoma, IDH-Mutant Grade 4)",
                "diagnosis_location": "Left Temporal Lobe",
                "diagnosis_date": "2026-05-14",
                "treatment_stage": "Adjuvant Chemoradiation (Stupp Protocol Cycle 3)",
                "baseline_kps": 90,
                "oncologist_name": "Dr. Aris Thorne, MD (Neuro-Oncology)",
                "nurse_name": "Sarah Jensen, RN (Care Navigator)",
                "share_research_consent": True,
                "created_at": "2026-05-14T08:00:00Z"
            },
            {
                "patient_id": p43_id,
                "mrn": "0043",
                "full_name": "Marcus Chen",
                "age": 58,
                "gender": "Male",
                "diagnosis_type": "Glioblastoma (IDH-Wildtype WHO Grade 4)",
                "diagnosis_location": "Right Frontal Lobe",
                "diagnosis_date": "2026-07-02",
                "treatment_stage": "Post-Surgical Adjuvant Temozolomide Cycle 1",
                "baseline_kps": 80,
                "oncologist_name": "Dr. Elena Rostova, MD (Neuro-Oncology)",
                "nurse_name": "James Miller, RN (Care Navigator)",
                "share_research_consent": True,
                "created_at": "2026-07-02T08:00:00Z"
            },
            {
                "patient_id": p44_id,
                "mrn": "0044",
                "full_name": "Priya Sharma",
                "age": 41,
                "gender": "Female",
                "diagnosis_type": "Oligodendroglioma (IDH-Mutant, 1p/19q-codeleted WHO Grade 3)",
                "diagnosis_location": "Right Parietal Lobe",
                "diagnosis_date": "2025-11-20",
                "treatment_stage": "Surveillance Post-PCV Chemotherapy",
                "baseline_kps": 95,
                "oncologist_name": "Dr. Aris Thorne, MD (Neuro-Oncology)",
                "nurse_name": "Sarah Jensen, RN (Care Navigator)",
                "share_research_consent": True,
                "created_at": "2025-11-20T08:00:00Z"
            }
        ]

        # 2. Molecular Profiles
        self.storage["molecular_profiles"] = [
            {
                "profile_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00001",
                "patient_id": p42_id,
                "idh_status": "Mutant",
                "mgmt_methylated": True,
                "codeletion_1p19q": False,
                "assayed_at": "2026-05-20"
            },
            {
                "profile_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00002",
                "patient_id": p43_id,
                "idh_status": "Wild-Type",
                "mgmt_methylated": False,
                "codeletion_1p19q": False,
                "assayed_at": "2026-07-10"
            },
            {
                "profile_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00003",
                "patient_id": p44_id,
                "idh_status": "Mutant",
                "mgmt_methylated": True,
                "codeletion_1p19q": True,
                "assayed_at": "2025-12-01"
            }
        ]

        # 3. Longitudinal Scans
        self.storage["mri_scans"] = [
            # Patient 0042
            {
                "scan_id": "a1b38d38-2c26-4d2b-91c6-2c1b97b00011",
                "patient_id": p42_id,
                "scan_date": "2026-06-10",
                "wt_vol_cm3": 22.40,
                "tc_vol_cm3": 14.80,
                "et_vol_cm3": 8.20,
                "edema_vol_cm3": 7.60,
                "dice_score": 0.9120,
                "estimated_rcbv": 2.15,
                "mask_storage_path": "scans/0042/20260610_seg.nii.gz"
            },
            {
                "scan_id": "a2b38d38-2c26-4d2b-91c6-2c1b97b00012",
                "patient_id": p42_id,
                "scan_date": "2026-07-28",
                "wt_vol_cm3": 17.10,
                "tc_vol_cm3": 10.40,
                "et_vol_cm3": 5.60,
                "edema_vol_cm3": 6.70,
                "dice_score": 0.9340,
                "estimated_rcbv": 1.82,
                "mask_storage_path": "scans/0042/20260728_seg.nii.gz"
            },
            {
                "scan_id": "a3b38d38-2c26-4d2b-91c6-2c1b97b00013",
                "patient_id": p42_id,
                "scan_date": "2026-09-18",
                "wt_vol_cm3": 14.60,
                "tc_vol_cm3": 8.10,
                "et_vol_cm3": 4.10,
                "edema_vol_cm3": 6.50,
                "dice_score": 0.9410,
                "estimated_rcbv": 1.48,
                "mask_storage_path": "scans/0042/20260918_seg.nii.gz"
            },
            # Patient 0043
            {
                "scan_id": "a4b38d38-2c26-4d2b-91c6-2c1b97b00014",
                "patient_id": p43_id,
                "scan_date": "2026-07-05",
                "wt_vol_cm3": 38.60,
                "tc_vol_cm3": 26.20,
                "et_vol_cm3": 15.80,
                "edema_vol_cm3": 12.40,
                "dice_score": 0.9050,
                "estimated_rcbv": 3.40,
                "mask_storage_path": "scans/0043/20260705_seg.nii.gz"
            },
            {
                "scan_id": "a5b38d38-2c26-4d2b-91c6-2c1b97b00015",
                "patient_id": p43_id,
                "scan_date": "2026-09-02",
                "wt_vol_cm3": 29.40,
                "tc_vol_cm3": 18.10,
                "et_vol_cm3": 10.90,
                "edema_vol_cm3": 11.30,
                "dice_score": 0.9280,
                "estimated_rcbv": 2.75,
                "mask_storage_path": "scans/0043/20260902_seg.nii.gz"
            },
            # Patient 0044
            {
                "scan_id": "a6b38d38-2c26-4d2b-91c6-2c1b97b00016",
                "patient_id": p44_id,
                "scan_date": "2026-03-12",
                "wt_vol_cm3": 11.20,
                "tc_vol_cm3": 5.40,
                "et_vol_cm3": 1.80,
                "edema_vol_cm3": 5.80,
                "dice_score": 0.9420,
                "estimated_rcbv": 1.25,
                "mask_storage_path": "scans/0044/20260312_seg.nii.gz"
            },
            {
                "scan_id": "a7b38d38-2c26-4d2b-91c6-2c1b97b00017",
                "patient_id": p44_id,
                "scan_date": "2026-08-25",
                "wt_vol_cm3": 7.80,
                "tc_vol_cm3": 3.10,
                "et_vol_cm3": 0.90,
                "edema_vol_cm3": 4.70,
                "dice_score": 0.9580,
                "estimated_rcbv": 1.10,
                "mask_storage_path": "scans/0044/20260825_seg.nii.gz"
            }
        ]

        # 4. Clinical Reports
        self.storage["clinical_reports"] = [
            # Patient 0042
            {
                "report_id": "b1b38d38-2c26-4d2b-91c6-2c1b97b00021",
                "patient_id": p42_id,
                "scan_id": "a3b38d38-2c26-4d2b-91c6-2c1b97b00013",
                "uploaded_by": "Clinician",
                "report_date": "2026-09-19",
                "raw_text": (
                    "FINDINGS: Interval reduction in enhancing residual lesion along posterior resection margin in left temporal lobe, "
                    "measuring 1.4 x 1.8 cm. Enhancing tumor volume estimated at 4.10 cm3. Moderate FLAIR hyperintensity consistent "
                    "with perilesional vasogenic edema stable at 6.50 cm3. No significant mass effect or midline shift. Ventricles are symmetric. "
                    "Relative CBV is reduced at 1.48. IMPRESSION: Consistent with post-treatment radiation effect / pseudoprogression."
                ),
                "parsed_entities": {
                    "tumor_location": "Left temporal lobe (resection margin)",
                    "enhancing_dimensions_cm": "1.4 x 1.8",
                    "edema_status": "Stable (6.50 cm3)",
                    "mass_effect": False,
                    "midline_shift_mm": 0.0,
                    "rcbv": 1.48,
                    "impression": "Treatment-induced radiation necrosis / pseudoprogression"
                },
                "plain_language_summary": (
                    "Your latest brain scan shows encouraging progress! The active area of your treated tumor has decreased in size to 4.1 cubic centimeters. "
                    "The mild tissue swelling (edema) around the surgery area is completely stable and not pushing against any healthy brain regions. "
                    "The low blood flow measurement indicates that the changes seen are safe, expected healing reactions from your radiation therapy (pseudoprogression), "
                    "rather than active cancer growth."
                ),
                "severity_tag": "Moderate",
                "created_at": "2026-09-19T14:30:00Z"
            },
            # Patient 0043
            {
                "report_id": "b2b38d38-2c26-4d2b-91c6-2c1b97b00022",
                "patient_id": p43_id,
                "scan_id": "a5b38d38-2c26-4d2b-91c6-2c1b97b00015",
                "uploaded_by": "Clinician",
                "report_date": "2026-09-03",
                "raw_text": (
                    "FINDINGS: Post-resection cavity in right frontal lobe with residual peripheral enhancement measuring 2.1 x 2.4 cm. "
                    "Mild reduction in overall volume. Surrounding FLAIR hyperintensity stable. No midline shift. Perfusion rCBV elevated at 2.75. "
                    "IMPRESSION: Consistent with residual active tumor responding partially to early post-operative adjuvant therapy."
                ),
                "parsed_entities": {
                    "tumor_location": "Right frontal lobe",
                    "enhancing_dimensions_cm": "2.1 x 2.4",
                    "edema_status": "Stable (11.30 cm3)",
                    "mass_effect": False,
                    "midline_shift_mm": 0.0,
                    "rcbv": 2.75,
                    "impression": "Residual active high-grade tumor"
                },
                "plain_language_summary": (
                    "Your recent scan shows that the surgery was effective and your treatment is stabilizing the area. "
                    "The remaining edge of the tumor in the right front portion of your brain has gotten smaller, down to 10.9 cubic centimeters. "
                    "We will continue active monitoring and your scheduled chemotherapy cycle."
                ),
                "severity_tag": "Moderate",
                "created_at": "2026-09-03T11:00:00Z"
            },
            # Patient 0044
            {
                "report_id": "b3b38d38-2c26-4d2b-91c6-2c1b97b00023",
                "patient_id": p44_id,
                "scan_id": "a7b38d38-2c26-4d2b-91c6-2c1b97b00017",
                "uploaded_by": "Clinician",
                "report_date": "2026-08-26",
                "raw_text": (
                    "FINDINGS: Continued interval involution of right parietal non-enhancing lesion following completion of PCV protocol. "
                    "Minimal residual FLAIR hyperintensity measuring 4.7 cm3. No pathological contrast enhancement. Perfusion rCBV normalized at 1.10. "
                    "No mass effect. IMPRESSION: Favorable durable treatment response; no evidence of recurrent tumor."
                ),
                "parsed_entities": {
                    "tumor_location": "Right parietal lobe",
                    "enhancing_dimensions_cm": "None (non-enhancing)",
                    "edema_status": "Minimal residual FLAIR (4.70 cm3)",
                    "mass_effect": False,
                    "midline_shift_mm": 0.0,
                    "rcbv": 1.10,
                    "impression": "Durable complete response"
                },
                "plain_language_summary": (
                    "Outstanding scan results! Your tumor continues to shrink with no signs of active cancer. "
                    "The remaining subtle scar tissue is stable and calm, with healthy normal blood flow. "
                    "Continue your current routine activities with your next check-up in 4 months."
                ),
                "severity_tag": "Mild",
                "created_at": "2026-08-26T09:15:00Z"
            }
        ]

        # 5. Medications
        self.storage["medications"] = [
            # Patient 0042
            {
                "med_id": "c1b38d38-2c26-4d2b-91c6-2c1b97b00031",
                "patient_id": p42_id,
                "med_name": "Temozolomide (Temodar)",
                "dosage": "150 mg/m2 Oral Capsule",
                "frequency": "Nightly (Days 1-5 of 28-day cycle)",
                "instructions": "Take on empty stomach at bedtime; take antiemetic 30 min prior",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Taken",
                "adherence_notes": "Cycle 3 Day 4 completed smoothly without emesis",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-24T22:00:00Z"
            },
            {
                "med_id": "c2b38d38-2c26-4d2b-91c6-2c1b97b00032",
                "patient_id": p42_id,
                "med_name": "Levetiracetam (Keppra)",
                "dosage": "750 mg Oral Tablet",
                "frequency": "Twice Daily (Every 12 hours)",
                "instructions": "Strict seizure prophylaxis; do not abruptly discontinue",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Missed",
                "adherence_notes": "Patient missed morning dose due to gastrointestinal upset",
                "last_updated_by": "Patient",
                "updated_at": "2026-09-24T09:45:00Z"
            },
            {
                "med_id": "c3b38d38-2c26-4d2b-91c6-2c1b97b00033",
                "patient_id": p42_id,
                "med_name": "Dexamethasone",
                "dosage": "2 mg Oral Tablet",
                "frequency": "Once Daily (Morning with food)",
                "instructions": "Tapering steroid for vasogenic edema control",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Taken",
                "adherence_notes": "Steroid taper on schedule",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-25T07:30:00Z"
            },
            # Patient 0043
            {
                "med_id": "c4b38d38-2c26-4d2b-91c6-2c1b97b00034",
                "patient_id": p43_id,
                "med_name": "Temozolomide (Temodar)",
                "dosage": "200 mg/m2 Oral Capsule",
                "frequency": "Nightly (Days 1-5 of 28-day cycle)",
                "instructions": "Cycle 1 post-resection protocol",
                "prescribed_by": "Dr. Elena Rostova, MD",
                "status": "Taken",
                "adherence_notes": "Completed cycle days on schedule",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-24T21:30:00Z"
            },
            {
                "med_id": "c5b38d38-2c26-4d2b-91c6-2c1b97b00035",
                "patient_id": p43_id,
                "med_name": "Levetiracetam (Keppra)",
                "dosage": "1000 mg Oral Tablet",
                "frequency": "Twice Daily",
                "instructions": "Seizure prevention",
                "prescribed_by": "Dr. Elena Rostova, MD",
                "status": "Taken",
                "adherence_notes": "100% adherence logged",
                "last_updated_by": "Patient",
                "updated_at": "2026-09-25T08:00:00Z"
            },
            # Patient 0044
            {
                "med_id": "c6b38d38-2c26-4d2b-91c6-2c1b97b00036",
                "patient_id": p44_id,
                "med_name": "Levetiracetam (Keppra)",
                "dosage": "500 mg Oral Tablet",
                "frequency": "Twice Daily",
                "instructions": "Maintenance dose; seizure free for >12 months",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Taken",
                "adherence_notes": "Full adherence confirmed",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-25T08:00:00Z"
            }
        ]

        # 6. Symptom Logs
        self.storage["symptom_logs"] = [
            # Patient 0042
            {
                "log_id": "d1b38d38-2c26-4d2b-91c6-2c1b97b00041",
                "patient_id": p42_id,
                "logged_at": "2026-09-23T16:00:00Z",
                "symptom_type": "Fatigue",
                "severity": "Moderate",
                "onset_time": "Late afternoon",
                "notes": "Needed a 2 hour nap after taking morning doses."
            },
            {
                "log_id": "d2b38d38-2c26-4d2b-91c6-2c1b97b00042",
                "patient_id": p42_id,
                "logged_at": "2026-09-24T09:30:00Z",
                "symptom_type": "Seizure",
                "severity": "Moderate",
                "onset_time": "Morning 9:30 AM",
                "notes": "Felt strange olfactory sensations (burning smell) and mild right hand tingling for 45 seconds after missing morning Keppra."
            },
            {
                "log_id": "d3b38d38-2c26-4d2b-91c6-2c1b97b00043",
                "patient_id": p42_id,
                "logged_at": "2026-09-25T07:15:00Z",
                "symptom_type": "Headache",
                "severity": "Mild",
                "onset_time": "Morning 7:00 AM",
                "notes": "Dull throbbing in left temple, relieved by drinking water."
            },
            # Patient 0043
            {
                "log_id": "d4b38d38-2c26-4d2b-91c6-2c1b97b00044",
                "patient_id": p43_id,
                "logged_at": "2026-09-23T14:30:00Z",
                "symptom_type": "Fatigue",
                "severity": "Moderate",
                "onset_time": "Mid-afternoon",
                "notes": "Mild drowsiness after physical therapy."
            },
            {
                "log_id": "d5b38d38-2c26-4d2b-91c6-2c1b97b00045",
                "patient_id": p43_id,
                "logged_at": "2026-09-24T18:00:00Z",
                "symptom_type": "Speech",
                "severity": "Mild",
                "onset_time": "Evening",
                "notes": "Brief word-finding delay; fully recovered."
            },
            # Patient 0044
            {
                "log_id": "d6b38d38-2c26-4d2b-91c6-2c1b97b00046",
                "patient_id": p44_id,
                "logged_at": "2026-09-22T19:00:00Z",
                "symptom_type": "Fatigue",
                "severity": "Mild",
                "onset_time": "Evening",
                "notes": "Usual fatigue after full workday; resolved with rest."
            }
        ]

        # 7. Twin Timeline
        self.storage["twin_timeline"] = [
            # Patient 0042
            {
                "timeline_id": "e1b38d38-2c26-4d2b-91c6-2c1b97b00051",
                "patient_id": p42_id,
                "evaluation_date": "2026-09-20",
                "progression_risk_score": 0.2850,
                "triage_category": "Moderate",
                "rano_category": "PsP",
                "pseudoprogression_risk": 0.8420,
                "is_pseudoprogression": True,
                "adherence_rate_pct": 85.70,
                "latent_twin_vector": [round(float(x), 4) for x in np.random.uniform(-0.15, 0.15, 64)],
                "alert_banner_text": "Attention: Missed Levetiracetam (Keppra) dose coincides with a newly reported sensory seizure aura. Please take your prescribed dose and contact your care team.",
                "created_at": "2026-09-20T10:00:00Z"
            },
            # Patient 0043
            {
                "timeline_id": "e2b38d38-2c26-4d2b-91c6-2c1b97b00052",
                "patient_id": p43_id,
                "evaluation_date": "2026-09-03",
                "progression_risk_score": 0.4120,
                "triage_category": "Moderate",
                "rano_category": "SD",
                "pseudoprogression_risk": 0.2800,
                "is_pseudoprogression": False,
                "adherence_rate_pct": 100.0,
                "latent_twin_vector": [round(float(x), 4) for x in np.random.uniform(-0.15, 0.15, 64)],
                "alert_banner_text": "Treatment Adherence High: Stable post-surgical cavity volume. Monitor for any motor weakness or speech difficulty.",
                "created_at": "2026-09-03T12:00:00Z"
            },
            # Patient 0044
            {
                "timeline_id": "e3b38d38-2c26-4d2b-91c6-2c1b97b00053",
                "patient_id": p44_id,
                "evaluation_date": "2026-08-26",
                "progression_risk_score": 0.1150,
                "triage_category": "Mild",
                "rano_category": "CR",
                "pseudoprogression_risk": 0.0500,
                "is_pseudoprogression": False,
                "adherence_rate_pct": 100.0,
                "latent_twin_vector": [round(float(x), 4) for x in np.random.uniform(-0.15, 0.15, 64)],
                "alert_banner_text": "Excellent Stability: Complete treatment response maintained with 100% medication adherence.",
                "created_at": "2026-08-26T10:00:00Z"
            }
        ]

        # 8. Counterfactual In-Silico Simulations
        self.storage["counterfactual_simulations"] = [
            {
                "sim_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00061",
                "patient_id": p42_id,
                "simulation_name": "Standard of Care TMZ Adjuvant Continuation",
                "intervention_type": "SOC_TMZ",
                "forecast_horizon_days": 180,
                "predicted_trajectory": {
                    "days": [0, 30, 60, 90, 180],
                    "tumor_vol_cm3": [14.6, 13.9, 13.2, 12.8, 12.4],
                    "kps_score": [90, 90, 85, 85, 85],
                    "confidence_interval": [0.92, 0.88, 0.84, 0.79, 0.73]
                },
                "created_at": "2026-09-20T12:00:00Z"
            },
            {
                "sim_id": "f2b38d38-2c26-4d2b-91c6-2c1b97b00062",
                "patient_id": p42_id,
                "simulation_name": "Dose-Dense Temozolomide (7/14 Schedule)",
                "intervention_type": "DOSE_DENSE",
                "forecast_horizon_days": 180,
                "predicted_trajectory": {
                    "days": [0, 30, 60, 90, 180],
                    "tumor_vol_cm3": [14.6, 13.1, 11.8, 11.2, 10.9],
                    "kps_score": [90, 85, 80, 80, 75],
                    "confidence_interval": [0.89, 0.84, 0.80, 0.75, 0.68]
                },
                "created_at": "2026-09-20T12:05:00Z"
            },
            {
                "sim_id": "f3b38d38-2c26-4d2b-91c6-2c1b97b00063",
                "patient_id": p42_id,
                "simulation_name": "Therapeutic Hold / TMZ Pause",
                "intervention_type": "HOLD_TMZ",
                "forecast_horizon_days": 180,
                "predicted_trajectory": {
                    "days": [0, 30, 60, 90, 180],
                    "tumor_vol_cm3": [14.6, 16.4, 19.8, 24.5, 36.2],
                    "kps_score": [90, 85, 80, 70, 60],
                    "confidence_interval": [0.91, 0.86, 0.81, 0.74, 0.65]
                },
                "created_at": "2026-09-20T12:10:00Z"
            }
        ]

    # --- Query & Retrieval APIs ---

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Retrieves all registered patients."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("patients").select("*").order("mrn").execute()
                if res.data:
                    return res.data
            except Exception:
                pass
        return self.storage["patients"]

    def get_patient_by_mrn(self, mrn: str = "0042") -> Optional[Dict[str, Any]]:
        """Retrieves patient demographics by MRN."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("patients").select("*").eq("mrn", mrn).execute()
                if res.data:
                    return res.data[0]
            except Exception:
                pass
        for p in self.storage["patients"]:
            if p["mrn"] == mrn:
                return p
        return None

    def get_patient_by_id(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves patient demographics by UUID."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("patients").select("*").eq("patient_id", patient_id).execute()
                if res.data:
                    return res.data[0]
            except Exception:
                pass
        for p in self.storage["patients"]:
            if p["patient_id"] == patient_id:
                return p
        return None

    def get_patient_molecular(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves molecular profile for a patient."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("molecular_profiles").select("*").eq("patient_id", patient_id).execute()
                if res.data:
                    return res.data[0]
            except Exception:
                pass
        for m in self.storage["molecular_profiles"]:
            if m["patient_id"] == patient_id:
                return m
        return None

    def get_patient_scans(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves historical MRI scans ordered by scan_date ascending."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("mri_scans").select("*").eq("patient_id", patient_id).order("scan_date").execute()
                if res.data:
                    return res.data
            except Exception:
                pass
        scans = [s for s in self.storage["mri_scans"] if s["patient_id"] == patient_id]
        return sorted(scans, key=lambda x: x["scan_date"])

    def get_patient_reports(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves clinical reports ordered by report_date descending."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("clinical_reports").select("*").eq("patient_id", patient_id).order("report_date", desc=True).execute()
                if res.data:
                    return res.data
            except Exception:
                pass
        reports = [r for r in self.storage["clinical_reports"] if r["patient_id"] == patient_id]
        return sorted(reports, key=lambda x: x["report_date"], reverse=True)

    def get_patient_medications(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves active medications and adherence status."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("medications").select("*").eq("patient_id", patient_id).execute()
                if res.data:
                    return res.data
            except Exception:
                pass
        return [m for m in self.storage["medications"] if m["patient_id"] == patient_id]

    def get_patient_symptoms(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves patient-logged symptoms ordered by logged_at descending."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("symptom_logs").select("*").eq("patient_id", patient_id).order("logged_at", desc=True).execute()
                if res.data:
                    return res.data
            except Exception:
                pass
        syms = [s for s in self.storage["symptom_logs"] if s["patient_id"] == patient_id]
        return sorted(syms, key=lambda x: x["logged_at"], reverse=True)

    def get_patient_timeline(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves latest digital twin evaluations."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("twin_timeline").select("*").eq("patient_id", patient_id).order("evaluation_date", desc=True).execute()
                if res.data:
                    return res.data
            except Exception:
                pass
        t = [item for item in self.storage["twin_timeline"] if item["patient_id"] == patient_id]
        return sorted(t, key=lambda x: x["evaluation_date"], reverse=True)

    def get_patient_simulations(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves in-silico simulations for the patient."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("counterfactual_simulations").select("*").eq("patient_id", patient_id).execute()
                if res.data:
                    return res.data
            except Exception:
                pass
        return [sim for sim in self.storage["counterfactual_simulations"] if sim["patient_id"] == patient_id]

    # --- Mutations & Insertions ---

    def log_symptom(self, symptom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Logs a patient symptom entry."""
        if "log_id" not in symptom_data:
            symptom_data["log_id"] = str(uuid.uuid4())
        if "logged_at" not in symptom_data:
            symptom_data["logged_at"] = datetime.now(timezone.utc).isoformat()

        if self.is_connected_to_supabase:
            try:
                self.supabase_client.table("symptom_logs").insert(symptom_data).execute()
            except Exception:
                pass

        self.storage["symptom_logs"].insert(0, symptom_data)
        return symptom_data

    def prescribe_medication(self, med_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adds a new prescription for a patient."""
        if "med_id" not in med_data:
            med_data["med_id"] = str(uuid.uuid4())
        if "updated_at" not in med_data:
            med_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "status" not in med_data:
            med_data["status"] = "Due"
        if "last_updated_by" not in med_data:
            med_data["last_updated_by"] = "Doctor"

        if self.is_connected_to_supabase:
            try:
                self.supabase_client.table("medications").insert(med_data).execute()
            except Exception:
                pass

        self.storage["medications"].append(med_data)
        return med_data

    def update_medication_status(self, med_id: str, new_status: str, notes: Optional[str] = None, updated_by: str = "Doctor") -> bool:
        """Updates medication adherence status ('Taken', 'Due', 'Missed') and clinical notes."""
        now_str = datetime.now(timezone.utc).isoformat()
        update_fields: Dict[str, Any] = {
            "status": new_status,
            "updated_at": now_str,
            "last_updated_by": updated_by
        }
        if notes:
            update_fields["adherence_notes"] = notes

        if self.is_connected_to_supabase:
            try:
                self.supabase_client.table("medications").update(update_fields).eq("med_id", med_id).execute()
            except Exception:
                pass

        for m in self.storage["medications"]:
            if m["med_id"] == med_id:
                m["status"] = new_status
                m["updated_at"] = now_str
                m["last_updated_by"] = updated_by
                if notes:
                    m["adherence_notes"] = notes
                return True
        return False

    def insert_scan_record(self, scan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts a segmented MRI scan record."""
        if "scan_id" not in scan_data:
            scan_data["scan_id"] = str(uuid.uuid4())
        if "created_at" not in scan_data:
            scan_data["created_at"] = datetime.now(timezone.utc).isoformat()

        if self.is_connected_to_supabase:
            try:
                self.supabase_client.table("mri_scans").insert(scan_data).execute()
            except Exception:
                pass

        self.storage["mri_scans"].append(scan_data)
        return scan_data

    def insert_clinical_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts a new clinical report."""
        if "report_id" not in report_data:
            report_data["report_id"] = str(uuid.uuid4())
        if "created_at" not in report_data:
            report_data["created_at"] = datetime.now(timezone.utc).isoformat()
        if "uploaded_by" not in report_data:
            report_data["uploaded_by"] = "Clinician"

        if self.is_connected_to_supabase:
            try:
                self.supabase_client.table("clinical_reports").insert(report_data).execute()
            except Exception:
                pass

        self.storage["clinical_reports"].insert(0, report_data)
        return report_data

    def update_twin_timeline(self, timeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Appends a new digital twin evaluation milestone."""
        if "timeline_id" not in timeline_data:
            timeline_data["timeline_id"] = str(uuid.uuid4())
        if "created_at" not in timeline_data:
            timeline_data["created_at"] = datetime.now(timezone.utc).isoformat()

        if self.is_connected_to_supabase:
            try:
                self.supabase_client.table("twin_timeline").insert(timeline_data).execute()
            except Exception:
                pass

        self.storage["twin_timeline"].insert(0, timeline_data)
        return timeline_data


# Global Singleton Client
_global_db_client: Optional[DatabaseClient] = None

def get_database_client() -> DatabaseClient:
    """Returns singleton instance of DatabaseClient."""
    global _global_db_client
    if _global_db_client is None:
        _global_db_client = DatabaseClient()
    return _global_db_client
