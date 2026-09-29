"""
FastAPI Backend Server for Glioma Digital Twin System
Exposes clinical endpoints for MONAI SegResNet volumetric segmentation,
Bio_ClinicalBERT NLP translation, Fisher-Kolmogorov biophysical simulation,
RANO 2.0 triage guardrails, and twin-grounded Care Companion chat.
"""

import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..models.mri_segmenter import SegResNetSegmenter, SyntheticBraTSGenerator
from ..models.clinical_nlp import ClinicalNLPEngine
from ..models.biophysical_solver import FisherKolmogorovSolver
from ..models.triage_engine import RANO2TriageEngine
from ..models.multimodal_twin import MultimodalTwinFusion
from ..models.companion_agent import CareCompanionAgent
from ..utils.supabase_client import get_database_client

app = FastAPI(
    title="Glioma Multimodal Digital Twin & Clinical AI Companion API",
    description="Production-grade AI pipeline bridging SegResNet 3D segmentation, ClinicalBERT translation, Fisher-Kolmogorov PDE modeling, and patient twin care.",
    version="1.0.0",
)

# CORS middleware for Next.js / Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons for models and engines
db = get_database_client()
segmenter = SegResNetSegmenter()
nlp_engine = ClinicalNLPEngine(offline=True)
pde_solver = FisherKolmogorovSolver()
triage_engine = RANO2TriageEngine()
fusion_model = MultimodalTwinFusion()
companion_agent = CareCompanionAgent()


# --- Pydantic Request Models ---

class NLPTranslateRequest(BaseModel):
    raw_text: str = Field(..., description="Free-text clinical radiology report")
    patient_id: Optional[str] = "e5b38d38-2c26-4d2b-91c6-2c1b97b00042"

class CounterfactualRequest(BaseModel):
    patient_id: Optional[str] = "e5b38d38-2c26-4d2b-91c6-2c1b97b00042"
    initial_vol_cm3: float = Field(14.60, ge=0.1, le=150.0)
    baseline_kps: int = Field(90, ge=10, le=100)
    idh_status: str = Field("Mutant", description="'Mutant' or 'Wild-Type'")
    mgmt_methylated: bool = Field(True, description="MGMT promoter methylation status")
    intervention: str = Field("SOC_TMZ", description="'SOC_TMZ', 'DOSE_DENSE', 'HOLD_TMZ', or 'LOMUSTINE'")
    forecast_horizon_days: int = Field(180, ge=30, le=365)

class TriageRequest(BaseModel):
    current_et_vol: float
    prior_et_vol: Optional[float] = None
    baseline_et_vol: float = 8.20
    days_since_rt_completion: int = 45
    rcbv: float = 1.48
    mgmt_methylated: bool = True
    midline_shift_mm: float = 0.0
    symptom_severity: str = "Mild"
    missed_meds_flag: bool = False

class SymptomLogRequest(BaseModel):
    patient_id: str = "e5b38d38-2c26-4d2b-91c6-2c1b97b00042"
    symptom_type: str = Field(..., description="Headache, Vision, Seizure, Fatigue, Speech, Motor")
    severity: str = Field(..., description="Mild, Moderate, Severe")
    onset_time: str
    notes: Optional[str] = None

class MedicationUpdateRequest(BaseModel):
    med_id: str
    new_status: str = Field(..., description="'Taken', 'Due', 'Missed'")

class CompanionChatRequest(BaseModel):
    user_message: str
    patient_id: str = "e5b38d38-2c26-4d2b-91c6-2c1b97b00042"

class ConsentUpdateRequest(BaseModel):
    patient_id: str = "e5b38d38-2c26-4d2b-91c6-2c1b97b00042"
    share_research_consent: bool


# --- API Routes ---

@app.get("/")
@app.get("/api/health")
def health_check():
    """System health check and module capability matrix."""
    return {
        "status": "healthy",
        "system": "Glioma Multimodal Digital Twin & Care Companion",
        "monai_available": True,
        "database_connected": db.is_connected_to_supabase,
        "device": str(segmenter.device),
        "supported_interventions": ["SOC_TMZ", "DOSE_DENSE", "HOLD_TMZ", "LOMUSTINE"],
        "supported_symptoms": ["Headache", "Vision", "Seizure", "Fatigue", "Speech", "Motor"],
    }

@app.get("/api/patient/{mrn}")
def get_patient_profile(mrn: str = "0042"):
    """Fetches complete clinical record and digital twin state for a patient."""
    patient = db.get_patient_by_mrn(mrn)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient with MRN '{mrn}' not found.")
    
    p_id = patient["patient_id"]
    molecular = db.get_patient_molecular(p_id)
    scans = db.get_patient_scans(p_id)
    reports = db.get_patient_reports(p_id)
    meds = db.get_patient_medications(p_id)
    symptoms = db.get_patient_symptoms(p_id)
    timeline = db.get_patient_timeline(p_id)
    sims = db.get_patient_simulations(p_id)

    # Compute running medication adherence rate
    total_meds = len(meds)
    taken_meds = sum(1 for m in meds if m.get("status") == "Taken")
    adherence_pct = round((taken_meds / max(1, total_meds)) * 100.0, 1)

    return {
        "patient": patient,
        "molecular_profile": molecular,
        "scans": scans,
        "reports": reports,
        "medications": meds,
        "adherence_rate_pct": adherence_pct,
        "symptoms": symptoms,
        "twin_timeline": timeline,
        "simulations": sims,
    }

@app.post("/api/segment/synthetic")
def run_synthetic_segmentation():
    """Generates synthetic 4-channel BraTS 3D volume and executes SegResNet segmentation."""
    vol_4ch, seg_gt = SyntheticBraTSGenerator.generate_synthetic_volume(shape=(64, 64, 48))
    seg_results = segmenter.extract_latent_and_segment(vol_4ch)
    
    # Exclude large 3D numpy arrays from direct JSON response for network speed
    return {
        "status": "success",
        "volumes_cm3": seg_results["volumes_cm3"],
        "dice_score": seg_results["dice_score"],
        "latent_vector_dimension": len(seg_results["latent_vector"]),
        "latent_vector_preview": seg_results["latent_vector"][:8],
    }

@app.post("/api/nlp/translate")
def translate_clinical_report(req: NLPTranslateRequest):
    """Processes free-text report, extracts diagnostic entities, and translates to 6th-grade summary."""
    res = nlp_engine.analyze_report(req.raw_text)
    return res

@app.post("/api/simulate/counterfactual")
def run_counterfactual_simulation(req: CounterfactualRequest):
    """Executes 3D reaction-diffusion Fisher-Kolmogorov PDE simulation."""
    res = pde_solver.simulate_counterfactual(
        initial_vol_cm3=req.initial_vol_cm3,
        baseline_kps=req.baseline_kps,
        idh_status=req.idh_status,
        mgmt_methylated=req.mgmt_methylated,
        intervention=req.intervention,
        forecast_horizon_days=req.forecast_horizon_days
    )
    return res

@app.post("/api/triage/rano")
def evaluate_rano_triage(req: TriageRequest):
    """Evaluates RANO 2.0 response and pseudoprogression urgency triage."""
    res = triage_engine.evaluate_response(
        current_et_vol=req.current_et_vol,
        prior_et_vol=req.prior_et_vol,
        baseline_et_vol=req.baseline_et_vol,
        days_since_rt_completion=req.days_since_rt_completion,
        rcbv=req.rcbv,
        mgmt_methylated=req.mgmt_methylated,
        midline_shift_mm=req.midline_shift_mm,
        symptom_severity=req.symptom_severity,
        missed_meds_flag=req.missed_meds_flag
    )
    return res

@app.post("/api/symptoms/log")
def log_patient_symptom(req: SymptomLogRequest):
    """Logs a patient symptom entry to the database."""
    item = db.log_symptom(req.dict())
    return {"status": "success", "symptom_log": item}

@app.post("/api/medications/update")
def update_medication_status(req: MedicationUpdateRequest):
    """Updates medication adherence status ('Taken', 'Due', 'Missed')."""
    success = db.update_medication_status(req.med_id, req.new_status)
    if not success:
        raise HTTPException(status_code=404, detail="Medication ID not found.")
    return {"status": "success", "med_id": req.med_id, "new_status": req.new_status}

@app.post("/api/companion/chat")
def companion_chat(req: CompanionChatRequest):
    """Generates twin-grounded Care Companion response with safety guardrails."""
    res = companion_agent.generate_response(req.user_message, patient_id=req.patient_id)
    return res

@app.post("/api/patient/{patient_id}/consent")
def update_consent(patient_id: str, req: ConsentUpdateRequest):
    """Updates patient research data sharing consent."""
    success = db.update_research_consent(patient_id, req.share_research_consent)
    return {"status": "success", "patient_id": patient_id, "share_research_consent": req.share_research_consent}
