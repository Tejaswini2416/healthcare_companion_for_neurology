# Healthcare Companion: AI-Powered Cancer Care
### *Multimodal Deep Learning, 3D Brain Modeling & Patient Health Twin for Brain Cancer*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-2596be.svg)](https://monai.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o%20Integrated-412991.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## 🌟 Overview & Clinical Vision

**Healthcare Companion** is an enterprise-grade, multimodal digital twin platform engineered to optimize care trajectories for patients confronting high-grade brain tumors (glioblastomas and high-grade astrocytomas). The platform bridges clinical engineering—featuring MONAI 3D SegResNet volumetric segmentation, Bio_ClinicalBERT diagnostic NLP, Fisher-Kolmogorov reaction-diffusion biophysical PDE simulation, cross-attention representation learning, interactive 3D anatomical brain modeling, and GPT-4o conversational care reasoning—with role-tailored workstations for oncologists and accessible portals for patients and caregivers.

### Key Clinical Capabilities:
- **3D Anatomical Brain & Tumor Localization Engine**: Interactive 3D cortical surface reconstruction (left and right cerebral hemispheres with gyri and sulcal folds, cerebellum, and brainstem) with patient-specific tumor sub-regions (Enhancing Core [ET], Necrotic Core [TC], Perilesional Edema [ED]) situated in their diagnosed anatomical lobe.
- **Conversational Care Companion GPT Architecture**: Dual-engine AI conversational companion connected to OpenAI GPT-4o / GPT-4o-mini, Groq LLaMA-3.3 70B, and a built-in Generative Clinical Bio-LLM. Explains complex oncology simply (including a 10-year-old child adventure story analogy), provides holistic diet/exercise/driving/travel guidance, and safeguards against emergency red flags.
- **Attending Physician Report Writing & Logging Workstation**: Dedicated clinical report composition suite for doctors with automated pre-filled templates, real-time Bio_ClinicalBERT entity extraction, plain-language translation generation, RANO 2.0 triage classification, and direct persistence to the digital twin database.
- **Official Clinical PDF Generation Suite**: ReportLab-powered PDF generator producing formatted clinical consultation reports with institutional headers, 3D volumetric metrics, technical findings, patient-friendly summaries, and clinician electronic sign-off—downloadable directly by patients and doctors.
- **Strict Clinical Document Validation Gate**: Document ingestion filter that strictly verifies authentic Radiology (Brain MRI, CT, PET) or Pathology reports and rejects non-medical documents (certificates, diplomas, invoices, resumes) with explicit clinical guidance.
- **Multi-View MRI Evaluator**: Simultaneous side-by-side visualization of Original Brain MRI, Neural Attention Heatmaps, and Grad-CAM Anatomical Saliency Overlays with instant digital twin volumetric synchronization.
- **3D Volumetric Tumor Decomposition**: Isolates Necrotic Core (NCR), Peritumoral Edema (ED), and Enhancing Tumor (ET) across multi-sequence MRI.
- **RANO 2.0 Pseudoprogression (PsP) Guard**: Differentiates treatment-induced radiation necrosis from genuine tumor recurrence within the critical 12-week post-radiotherapy window using rCBV perfusion and MGMT promoter methylation status.
- **Biophysical In-Silico Forecasting**: Solves a 3D reaction-diffusion proliferation-invasion PDE to simulate counterfactual tumor volume and Karnofsky Performance Scale (KPS) trajectories over 30 to 180-day horizons across multiple therapeutic regimens.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Multi-Source Ingestion & Verification
        A[4-Channel 3D MRI: T1, T1ce, T2, FLAIR] --> B[MONAI SegResNet 3D]
        C[2D MRI Slices / Uploads] --> D[Deep ResNet-18 Classifier]
        E[Clinical Free Text / Reports] --> V_GATE[Strict Radiology/Pathology Gate]
        V_GATE --> F[Bio_ClinicalBERT NLP Engine]
        G[Structured EHR & Adherence] --> H[EHR Feature Vector]
    end

    subgraph Feature Representations & Volumetrics
        B -->|Deep Bottleneck GAP| I[128D MRI Latent Vector]
        B -->|Sub-Regions| J[NCR, ED, ET, WT, TC cm³]
        D -->|Grad-CAM| K[Side-by-Side Saliency Map + Confidence]
        F -->|L2 Normalized| L[11D Semantic Vector]
        F -->|Patient Translation| M[6th-Grade Plain Language]
        H --> N[8D Normalized EHR Vector]
    end

    subgraph Biophysical, Triage & 3D Reconstruction
        J --> O[Fisher-Kolmogorov PDE Solver]
        O -->|Counterfactuals| P[30-180d Vol & KPS Trajectories]
        J & F & G --> Q[RANO 2.0 & PsP Triage Guard]
        Q --> R[RANO Category & Triage Score]
        J & E --> BRAIN_3D[3D Anatomical Brain & Tumor Localization Engine]
    end

    subgraph Intermediate Attention Fusion
        I & L & N --> S[Cross-Attention PyTorch Module]
        S --> T[64D Unified Patient Health Twin]
        T --> U[Progression Risk & Modality Weights]
    end

    subgraph Clinical & Conversational Interfaces
        T & P & R & BRAIN_3D --> CLINICIAN[Doctor Workstation & Report Writer]
        CLINICIAN --> PDF_GEN[Official Clinical PDF Generator]
        T & M & K & BRAIN_3D --> PATIENT[Patient Portal & Scans Hub]
        PATIENT --> GPT_AGENT[Care Companion GPT-4o / Clinical Bio-LLM]
        PATIENT --> PDF_GEN
        CLINICIAN & PATIENT <--> DB[(Supabase PostgreSQL / Local Resilient DB)]
    end
```

---

## 🔬 Core Architectural Modules

### 1. 🧠 Interactive 3D Anatomical Brain & Tumor Localization (`brain_3d_viewer.py`)
- **Cortical Surface Modeling**: High-fidelity Plotly 3D parametric surface reconstruction representing the human brain cortex silhouette (left and right cerebral hemispheres with gyri and sulcal folds, cerebellum, and brainstem).
- **Stereotactic Sub-Region Localization**: Uses clinical anatomical lobe mapping to situate patient-specific tumor sub-regions inside the brain:
  - 🔴 **Enhancing Tumor Core (ET)**: Measured residual post-operative lesion rim.
  - 🟠 **Tumor Core / Necrosis (TC)**: Central hypo-intense necrotic core.
  - 🟡 **Vasogenic Peritumoral Edema (ED)**: Reactive fluid envelope surrounding the lesion.
- **Interactive Controls**: Full 360° orbital rotation, pan, zoom, cortical opacity transparency slider (0.05 to 0.60), and camera presets (*Perspective 3/4*, *Anterior*, *Superior*, *Left Lateral Profile*, *Right Lateral Profile*, *Posterior*).
- **Embedded Across**: Doctor Portal (dedicated view & Clinician Workstation), Patient Portal (*My Scans & Reports* and dedicated view), and the 2D MRI Evaluator.

### 2. 📝 Attending Physician Report Writer & Logging Console (`doctor_report_writer.py`)
- **Direct Database Logging**: Attending physicians and neuroradiologists can compose, customize, and log clinical reports directly into `db.insert_clinical_report` and update patient digital twin timeline milestones.
- **Clinical Templates**: Pre-configured standard scenarios for *Surveillance Brain MRI (Stable/PsP)*, *Recurrent Disease Progression*, and *Complete Response*.
- **Integrated Intelligence**: Automatic entity parsing via Bio_ClinicalBERT, automated plain-language patient translation generation, and RANO 2.0 triage classification.
- **One-Click PDF Export**: Immediately compiles and exports signed reports to PDF for medical records.

### 3. 📄 Official Clinical Report PDF Generation Suite (`pdf_generator.py`)
- **ReportLab Engine**: Compiles official, hospital-grade clinical consultation summaries formatted with professional styling.
- **Comprehensive Structure**:
  - Hospital Letterhead & Attending Physician Attestation.
  - Patient Demographics & Diagnostic Profile (MRN, DOB, primary diagnosis, location).
  - 3D Tumor Volumetric Table (Whole Tumor, Enhancing Core, Edema, rCBV perfusion).
  - Technical Narrative Findings & Impression.
  - Empathetic 6th-Grade Patient Plain-Language Translation.
  - Formal Electronic Clinician Signature block.
- **Patient Download**: Patients can download their official reports with a single click (*"📥 Download Official Clinical Report (PDF)"*) in the Patient Portal.

### 4. 🤖 Care Companion GPT Architecture (`companion_agent.py`, `companion_chat.py`)
- **Dual-Engine LLM Core**:
  - **OpenAI GPT-4o / GPT-4o-mini Integration**: Direct cloud inference with complete Patient Digital Twin context injection.
  - **Groq LLaMA-3.3 70B Integration**: High-speed, high-intelligence cloud LLM inference.
  - **Built-in Clinical Bio-LLM Engine**: Generates natural, empathetic, ChatGPT-quality responses even without an external API key.
- **Adaptive Explanations**:
  - *10-Year-Old Explanation*: Explains brain anatomy, surgery, and chemotherapy using superhero team and control center analogies.
  - *Diet & Nutrition*: Clinical guidance on protein intake, hydration, coffee/caffeine, and steroid-related sugar management.
  - *Daily Safety*: DMV/AAN driving intervals, air travel safety, and seizure precautions.
  - *Emotional Support*: Reassurance that brain cancer is non-contagious and compassionate counseling for treatment fatigue.
- **ChatGPT Interface**: Features model switcher, API key configuration card, suggested prompt chips, message reset (`🗑️ Reset Chat`), and model attribution badges.

### 5. 🛡️ Strict Clinical Document Verification Gate (`report_verification_hub.py`)
- **Radiology & Pathology Verification**: Enforces strict verification of clinical terminology (MRI sequences, DSC perfusion, rCBV, WHO grading, histopathology, margins).
- **Non-Medical Rejection**: Rejects non-clinical files (such as coursework certificates, diplomas, invoices, or resumes) with clear guidance.
- **Standardized Execution Button**: Standardized trigger button named strictly **"🚀 Run Uploaded Reports"**.

### 6. 🖼️ Multi-View MRI Saliency Evaluator (`mri_evaluator.py`)
- **Side-by-Side 3-View Layout**: Simultaneous visualization of:
  1. *Original Brain MRI*
  2. *Neural Attention Heatmap*
  3. *Grad-CAM Anatomical Saliency Overlay*
- **Instant Digital Twin Sync**: Button to persist evaluated scans directly to Supabase (`public.mri_scans`) and update digital twin timelines.
- **Embedded 3D Brain Viewer**: Displays the evaluated scan inside the 3D anatomical brain model immediately below diagnostic findings.

### 7. Medical Imaging & Volumetric Analysis (3D SegResNet & 2D ResNet-18)
- **3D SegResNet Architecture**: Built on MONAI (`spatial_dims=3, in_channels=4, out_channels=3`) processing 4-channel NIfTI volumes:
  - `Channel 0`: Necrotic and Non-enhancing Tumor Core (NCR)
  - `Channel 1`: Peritumoral Vasogenic Edema (ED)
  - `Channel 2`: Active Enhancing Tumor (ET)
  - `Derived`: Whole Tumor ($\text{WT} = \text{NCR} \cup \text{ED} \cup \text{ET}$) and Tumor Core ($\text{TC} = \text{NCR} \cup \text{ET}$).
- **Deep ResNet-18 2D Brain Tumor Classifier**:
  - Pretrained deep CNN fine-tuned on clinical brain tumor scans.
  - **Holdout Test Set Performance (38 unseen scans)**:
    - **Accuracy**: **89.47%**
    - **Sensitivity (Recall)**: **100.0%** (23/23 tumors detected, 0 False Negatives)
    - **Specificity**: **73.33%**
    - **ROC-AUC**: **0.9652** | **PR-AUC**: **0.9755**
  - **Grad-CAM**: Generates high-resolution saliency maps overlaid on scan slices for visual auditability.

### 8. Biophysical In-Silico Simulation (Fisher-Kolmogorov PDE)
- **Reaction-Diffusion Proliferation & Invasion Solver**:
  $$\frac{\partial c}{\partial t} = \nabla \cdot (D \nabla c) + \rho \cdot c (1 - c) - k_{\text{kill}} \cdot c$$
  - $c(r, t)$: Spatiotemporal tumor cell density.
  - $D$: Cell motility / diffusivity ($\text{cm}^2/\text{day}$).
  - $\rho$: Proliferation rate calibrated by IDH mutation status.
  - $k_{\text{kill}}$: Cytotoxic cell kill calibrated by MGMT promoter methylation.
- **Counterfactual Therapeutic Regimens**: Standard-of-Care TMZ, Dose-Dense, Treatment Hold, and Lomustine.
- **Predictive Trajectory**: Yields projected volumetric trajectories ($\text{cm}^3$) and Karnofsky Performance Scale ($\text{KPS}$) over 30 to 180 days.

### 9. RANO 2.0 Pseudoprogression (PsP) Guard & Urgency Triage
- **Temporal Rule**: Scans within 12 weeks post-radiotherapy trigger the PsP protocol.
- **Perfusion Calibration**: Evaluates relative Cerebral Blood Volume ($\text{rCBV}$) against a $1.75$ threshold and checks MGMT methylation.
- **Standardized Categories**: Outputs formal RANO categories (`CR`, `PR`, `SD`, `PD`, `PsP`) and clinical triage urgency scores (`Mild`, `Moderate`, `Critical`).

### 10. GraphRAG Clinical Safety Guard & Knowledge Graph (`graph_rag.py`)
- **Evidence Grounding**: Multi-hop traversal over NCCN Guidelines for Central Nervous System Cancers (v2026.1) and AAN Standards.
- **Anti-Hallucination Fact Verification**: Dynamically outputs verification confidence scores ($>90\%$) and expandable citation drawers.

---

## 🗄️ Database Architecture (Supabase PostgreSQL)

The schema is defined in [`database/supabase_schema.sql`](file:///c:/Users/Tejaswini/Desktop/digital_twin/database/supabase_schema.sql).

### Normalized Tables:
1. `patients`: UUID PK, MRN (unique), demographics, diagnosis location, baseline KPS, care team, and research consent flags.
2. `molecular_profiles`: IDH mutation status, MGMT promoter methylation, and 1p/19q codeletion.
3. `mri_scans`: Serial volumetric measurements ($\text{WT}$, $\text{TC}$, $\text{ET}$, $\text{Edema}$ in $\text{cm}^3$), Dice scores, and estimated $\text{rCBV}$.
4. `clinical_reports`: Uploads by doctors or patients, raw text, parsed JSONB entities, plain-language translations, and severity tags.
5. `medications`: Active prescriptions, dose, frequency, adherence status (`Taken`, `Due`, `Missed`), clinical adherence notes, and update audit trail.
6. `symptom_logs`: Patient entries categorized by symptom type, discrete severity (`Mild`, `Moderate`, `Severe`), onset timestamps, and notes.
7. `twin_timeline`: Longitudinal digital twin evaluations, progression risk scores, RANO 2.0 categories, PsP probability, 64D latent vector, and alert banners.
8. `counterfactual_simulations`: In-silico parameter calibrations, intervention types, and predicted 180-day trajectories.

### Pre-Seeded Patient Profiles:
- **Patient 0042: V. Thanuja** (Age 24, Left Temporal High-Grade Glioma, IDH-Mutant / MGMT-Methylated, Active Chemoradiation)
- **Patient 0043: Marcus Chen** (Age 58, Right Frontal Glioblastoma, IDH-Wildtype / MGMT-Unmethylated, Post-Surgical Adjuvant TMZ)
- **Patient 0044: Priya Sharma** (Age 41, Right Parietal Oligodendroglioma, IDH-Mutant / 1p19q-Codeleted, Surveillance Post-PCV)

---

## 🖥️ User Roles & Workflows

### 🩺 Clinician Workstation & Doctor Portal
- **Doctor Authentication & RBAC Gate**: Institutional login for attending specialists (*Dr. Aris Thorne*, *Dr. Elena Rostova*, *Dr. Marcus Vance*).
- **Report Writing & Logging Page**: Compose, template, verify, and log clinical reports directly into the database.
- **3D Brain Volumetric Reconstruction**: Inspect anatomical 3D brain surfaces with color-coded tumor sub-region localization.
- **Cohort Roster & Multi-Patient Inspector**: Detailed demographics, molecular profiles, and longitudinal metrics.
- **Diagnostic Ingestion Suite**: SegResNet 3D BraTS segmentation and 2D MRI ResNet-18 classifier with Grad-CAM overlays.
- **Medication Governance & In-Silico Horizon Simulator**: Prescribe regimens and run 180-day counterfactual trajectory simulations.

### 👤 Patient Portal
- **Home Dashboard**: Digital twin alert banner, stat cards (Tumor Volume, Medication Adherence, Symptoms, Next Visit), and navigation.
- **My Scans & Reports**: Technical findings vs. plain-language translations, interactive 3D Brain Viewer, and official clinical PDF download.
- **Conversational Care Companion GPT**: Twin-grounded conversational agent powered by GPT-4o / Clinical Bio-LLM with suggested prompt chips.
- **Quick Symptom Logger**: Fast one-tap symptom reporting (`[Headache]`, `[Vision]`, `[Seizure]`, `[Fatigue]`, `[Speech]`, `[Motor]`).
- **Medication Tracker**: Daily schedule with one-tap adherence buttons and live compliance rate tracking.
- **Upload & Verify Reports**: Strict clinical report verification hub with automated triage and 10-year-old audio translation.

---

## 📂 Repository Directory Layout

```
digital_twin/
├── database/
│   └── supabase_schema.sql           # Complete Supabase PostgreSQL DDL with RLS & pre-seeds
├── data/
│   ├── sample_reports/               # Standardized clinical consultation & radiology report library
│   │   ├── 01_pseudoprogression_radiation_necrosis.txt # Sample 1: PsP / radiation necrosis (MRN: 0042)
│   │   ├── 02_recurrent_high_grade_progression.txt     # Sample 2: True tumor recurrence (MRN: 0043)
│   │   ├── 03_complete_treatment_response.txt          # Sample 3: Complete durable response (MRN: 0044)
│   │   ├── 04_acute_mass_effect_herniation.txt         # Sample 4: Acute herniation emergency (STAT)
│   │   └── README.md                                   # Clinical parameters & test guide
│   └── sample_scans/                 # Clinical MRI scan dataset for 2D tumor classifier
│       ├── yes/                      # Tumor-positive scans
│       └── no/                       # Tumor-negative scans
├── glioma_digital_twin/
│   ├── models/
│   │   ├── mri_segmenter.py          # MONAI SegResNet 3D volumetric segmentation & BraTS generator
│   │   ├── clinical_nlp.py           # Bio_ClinicalBERT parser & multilingual patient translation
│   │   ├── biophysical_solver.py     # Fisher-Kolmogorov 3D reaction-diffusion PDE solver
│   │   ├── triage_engine.py          # RANO 2.0 pseudoprogression guard & triage engine
│   │   ├── multimodal_twin.py        # PyTorch intermediate cross-attention fusion module
│   │   ├── companion_agent.py        # GPT-4o & Clinical Bio-LLM companion agent with GraphRAG guard
│   │   ├── graph_rag.py              # Knowledge graph clinical guard & NCCN guidelines
│   │   ├── red_flag_alert.py         # STAT emergency escalation & triage dispatcher
│   │   ├── tumor_classifier.py       # Deep ResNet-18 inference wrapper with Grad-CAM
│   │   ├── train_tumor_model.py      # ResNet-18 training, validation, and benchmarking pipeline
│   │   └── saved_models/
│   │       ├── best_tumor_classifier.pt # Trained ResNet-18 model weights (42.9 MB)
│   │       ├── evaluation_results.json  # Comprehensive test evaluation metrics & benchmarks
│   │       ├── roc_pr_curves.png        # ROC and Precision-Recall curves
│   │       ├── training_curves.png      # Loss and accuracy training trajectories
│   │       ├── confusion_matrix.png     # Holdout test set confusion matrix
│   │       └── test_predictions_gradcam.png # Grad-CAM localization visual validation
│   ├── ui/
│   │   ├── app.py                    # Main Streamlit web application entry point & router
│   │   └── components/
│   │       ├── brain_3d_viewer.py     # Interactive 3D Anatomical Brain & Tumor Localization Engine
│   │       ├── doctor_report_writer.py # Doctor Clinical Report Writing & Logging Workstation
│   │       ├── doctor_login.py        # Doctor Authentication & Patient Cohort Inspector
│   │       ├── clinician_upload.py    # Clinician Workstation (Roster, 3D Scans, Meds, Report, Simulator)
│   │       ├── report_verification_hub.py # Strict Clinical Document Verification & Execution Hub
│   │       ├── patient_dashboard.py   # Patient Portal Home Dashboard & Stat Cards
│   │       ├── report_viewer.py       # Scans, plain translations, PDF download, and 3D Brain Viewer
│   │       ├── symptom_logger.py      # One-tap symptom entry form & timeline
│   │       ├── companion_chat.py      # Care Companion GPT chat & medication tracker
│   │       ├── longitudinal_trends.py # Volumetric, symptom, and adherence analytics
│   │       ├── volume_viewer_3d.py    # Three.js WebGL volumetric scan renderer
│   │       ├── audio_tts_player.py    # Edge-TTS voice-over audio player
│   │       └── mri_evaluator.py       # Side-by-side 3-view 2D MRI Evaluator with Grad-CAM
│   ├── utils/
│   │   ├── pdf_generator.py          # ReportLab official clinical report PDF generator
│   │   ├── supabase_client.py        # Dual-mode (Supabase + local persistent memory) client
│   │   └── synthetic_data.py         # Multi-sequence NIfTI BraTS volume generator
│   └── run_demo.py                   # Standalone pipeline verification script
├── run_demo.py                       # Root launcher for pipeline verification
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container deployment specification
├── README.md                         # Comprehensive technical documentation
└── package.json                      # Workspace configuration
```

---

## 🚀 Quickstart & Execution

### 1. Prerequisites
- Python 3.10 or 3.11
- CUDA-compatible GPU (optional, fully supports CPU execution)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Tejaswini2416/healthcare_companion_for_neurology.git
cd healthcare_companion_for_neurology

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Launch Interactive Web Application
```bash
streamlit run glioma_digital_twin/ui/app.py --server.port=8501
```
Open your browser and navigate to:
```
http://localhost:8501
```

### 4. Run End-to-End Pipeline Verification
```bash
python run_demo.py --no-launch
```

---

## 🔒 Security, Compliance & Ethical Disclaimer

> **IMPORTANT CLINICAL NOTICE**:
> **Healthcare Companion** is an investigational clinical research platform designed to assist medical professionals and enhance patient engagement. It is **not** an FDA-cleared independent medical diagnostic device. All treatment plans, surgical interventions, prescription modifications, and therapeutic halts must be evaluated and confirmed by licensed oncologists and multidisciplinary tumor boards.
>
> **Data Privacy & Governance**:
> Database access is governed by PostgreSQL Row-Level Security (RLS) policies. In production deployments with live Supabase instances, all PHI/PII data transmissions must comply with HIPAA, GDPR, and institutional IRB research standards.

---

## 📄 License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
