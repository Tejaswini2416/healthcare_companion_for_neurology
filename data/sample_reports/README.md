# Neuro-Oncology Clinical Sample Reports Library

This directory contains standardized clinical radiology and neuroradiology MRI consultation reports engineered for testing, demonstration, and AI validation within the **Healthcare Companion Clinician Workstation**.

## Available Sample Reports

| File | Scenario | Key Findings | Expected AI Severity | Expected RANO Category |
| :--- | :--- | :--- | :--- | :--- |
| [`01_pseudoprogression_radiation_necrosis.txt`](file:///c:/Users/Tejaswini/Desktop/digital_twin/data/sample_reports/01_pseudoprogression_radiation_necrosis.txt) | **Pseudoprogression (PsP) / Radiation Effect** | Left temporal lobe, rCBV = 1.45, decreased enhancement, 0 mm midline shift, stable edema | **Mild / Favorable** | **PsP (Pseudoprogression >85%)** |
| [`02_recurrent_high_grade_progression.txt`](file:///c:/Users/Tejaswini/Desktop/digital_twin/data/sample_reports/02_recurrent_high_grade_progression.txt) | **True High-Grade Tumor Recurrence** | Right frontal lobe, marked rim enhancement, elevated rCBV = 3.65, 3.5 mm midline shift, 19.8 cm³ edema | **Critical / Urgent** | **PD (Progressive Disease)** |
| [`03_complete_treatment_response.txt`](file:///c:/Users/Tejaswini/Desktop/digital_twin/data/sample_reports/03_complete_treatment_response.txt) | **Durable Complete Treatment Response** | Right parietal lobe, non-enhancing, normalized rCBV = 1.10, 0 mm midline shift, no mass effect | **Mild / Reassuring** | **CR (Complete Response)** |
| [`04_acute_mass_effect_herniation.txt`](file:///c:/Users/Tejaswini/Desktop/digital_twin/data/sample_reports/04_acute_mass_effect_herniation.txt) | **Acute Mass Effect & Herniation Alert** | Severe edema 24.5 cm³, 5.8 mm subfalcine midline shift, elevated rCBV = 2.95, impending herniation | **Critical / STAT** | **Critical Urgent Triage** |

## How to Test in the Clinician Workstation:
1. Navigate to **Clinician Workstation** (Login with Doctor credentials or 1-Click Doctor Quick Login).
2. Go to the **"📄 Clinical Report Upload & AI Verification Hub"** tab.
3. You can either:
   - **Upload directly**: Drag-and-drop any of the `.txt` files above.
   - **1-Click Quick Select**: Select any sample report from the interactive dropdown to immediately stage and test.
4. Click **"🚀 Run Bio_ClinicalBERT Extraction & Verification Engine"** to verify:
   - Extracted diagnostic entities (tumor location, margins, rCBV, midline shift, edema).
   - Real-time 6th-grade patient translation.
   - 11-dimensional semantic embedding vector.
   - Clinical triage priority score.
5. Review clinical findings, add doctor verification notes, and click **"✅ Verify & Commit Report to Patient Digital Twin"**.
