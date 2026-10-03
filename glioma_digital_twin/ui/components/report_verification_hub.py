"""
Clinical Radiology Report Ingestion & Verification Hub Component
Enables:
1. Multi-scenario sample report library with 1-click loading and file downloads.
2. File drag-and-drop (.txt, .md, .json, .pdf text streams) and free-text narrative editing.
3. Bio_ClinicalBERT entity parsing, rCBV perfusion thresholding, and RANO 2.0 triage classification.
4. 6th-grade reading level empathetic patient translation generation.
5. Attending physician verification, electronic sign-off, and real-time synchronization with the Patient Digital Twin.
"""

import os
import uuid
import hashlib
import streamlit as st
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from ...utils.supabase_client import get_database_client
from ...models.clinical_nlp import ClinicalNLPEngine
from ...models.triage_engine import RANO2TriageEngine


SAMPLE_REPORTS_LIBRARY = [
    {
        "id": "sample_01_psp",
        "title": "Sample 1: Treatment-Induced Pseudoprogression (PsP)",
        "scenario": "Radiation Necrosis / Pseudoprogression",
        "mrn": "0042",
        "target_patient": "V. Thanuja (MRN: 0042)",
        "file_name": "01_pseudoprogression_radiation_necrosis.txt",
        "badge_color": "#10b981",
        "expected_severity": "Mild / Favorable",
        "expected_rano": "PsP (Pseudoprogression Probability >85%)",
        "summary": "Left temporal lobe, rCBV = 1.45, decreased enhancement, stable edema (6.3 cm³), 0.0 mm midline shift.",
        "text": """CLINICAL RADIOLOGY & MULTI-PARAMETRIC MRI REPORT
PATIENT: V. Thanuja
MRN: 0042
DATE OF EXAMINATION: 2026-09-28
CLINICAL INDICATION: 24-year-old female with IDH-mutant Grade 4 astrocytoma of the left temporal lobe, post-surgical resection, undergoing adjuvant Stupp protocol temozolomide chemoradiation. Surveillance MRI evaluation.

TECHNIQUE: Multi-planar, multi-sequence brain MRI performed with and without intravenous gadolinium contrast (0.1 mmol/kg Gadavist). Sequences include axial T1-weighted, T2-weighted, 3D FLAIR, dynamic susceptibility contrast (DSC) perfusion, and axial/coronal/sagittal T1 post-contrast.

FINDINGS:
BRAIN PARENCHYMA & RESECTION CAVITY:
Stable surgical resection cavity in the left temporal lobe. Along the posterior margin of the cavity, there is an enhancing residual lesion measuring 1.3 x 1.7 cm, demonstrating an estimated enhancing tumor volume of 3.85 cm3. In comparison to the examination from 2026-07-28, there is interval reduction in enhancing volume and margin definition.

PERITUMORAL VASOGENIC EDEMA & MASS EFFECT:
Surrounding FLAIR hyperintensity in the peritumoral white matter is stable at 6.30 cm3, representing mild localized edema. There is no significant mass effect. No effacement of the temporal horn of the left lateral ventricle. Midline structures are entirely symmetric with 0.0 mm midline shift.

DSC PERFUSION MRI:
Dynamic susceptibility contrast perfusion demonstrates low relative cerebral blood volume within the enhancing rim, with mean rCBV measured at 1.45 (relative to normal contralateral white matter). No areas of markedly elevated microvascular perfusion or neoangiogenesis.

DIFFUSION-WEIGHTED IMAGING (DWI):
No restricted diffusion to suggest acute ischemia or hypercellular tumor nidus.

IMPRESSION:
1. Interval decrease in enhancing residual volume along the left temporal resection cavity margin with stable surrounding FLAIR edema.
2. Low perfusion rCBV (1.45) in conjunction with recent completion of radiation therapy and MGMT methylated promoter status is strongly indicative of treatment-induced pseudoprogression / radiation necrosis (RANO PsP probability >85%) rather than true tumor recurrence.
3. Stable clinical trajectory without mass effect or midline shift. Continue current adjuvant temozolomide therapy."""
    },
    {
        "id": "sample_02_progression",
        "title": "Sample 2: True High-Grade Recurrence / Progression",
        "scenario": "Glioblastoma True Neoplastic Progression",
        "mrn": "0043",
        "target_patient": "Marcus Chen (MRN: 0043)",
        "file_name": "02_recurrent_high_grade_progression.txt",
        "badge_color": "#ef4444",
        "expected_severity": "Critical / Urgent",
        "expected_rano": "PD (Progressive Disease)",
        "summary": "Right frontal lobe, rCBV = 3.65, marked rim enhancement, 3.5 mm midline shift, 19.8 cm³ edema.",
        "text": """CLINICAL RADIOLOGY & MULTI-PARAMETRIC MRI REPORT
PATIENT: Marcus Chen
MRN: 0043
DATE OF EXAMINATION: 2026-09-29
CLINICAL INDICATION: 58-year-old male with glioblastoma (IDH-wildtype, MGMT-unmethylated, WHO Grade 4) of the right frontal lobe, status post gross total resection and adjuvant chemoradiation. New onset worsening left-sided pronator drift and cognitive slowing.

TECHNIQUE: 3D Multi-parametric brain MRI with DSC perfusion and diffusion tensor imaging before and after 15 mL Dotarem IV contrast.

FINDINGS:
BRAIN PARENCHYMA & RESECTION CAVITY:
Evaluation of the right frontal lobe resection bed demonstrates marked interval progression. A thick, irregular, nodular rim of intense contrast enhancement now surrounds the lateral and deep aspects of the surgical cavity, measuring 3.4 x 2.8 cm, with estimated enhancing tumor volume increased to 18.40 cm3 (previously 10.90 cm3).

PERITUMORAL EDEMA & MASS EFFECT:
There is extensive expanding perilesional vasogenic edema throughout the right frontal white matter and corona radiata, now measuring 19.80 cm3. Moderate mass effect is noted with partial effacement of the right lateral ventricle frontal horn and 3.5 mm leftward midline shift.

DSC PERFUSION ANALYSIS:
Dynamic susceptibility contrast-derived perfusion demonstrates intense hypervascularity and microvascular proliferation within the nodular enhancing regions, with peak rCBV measuring 3.65 (reference range <1.75).

DIFFUSION-WEIGHTED IMAGING:
Focal areas of restricted diffusion (ADC values < 750 x 10^-6 mm2/s) within the enhancing nodularity consistent with high cellular density.

IMPRESSION:
1. Unequivocal true neoplastic tumor recurrence and progressive disease (RANO 2.0 Category: Progressive Disease / PD) within the right frontal lobe resection margin.
2. Marked elevation of relative cerebral blood volume (rCBV 3.65) and restricted diffusion confirm aggressive active tumor neoangiogenesis.
3. Moderate mass effect with 3.5 mm midline shift. Urgent neuro-oncology tumor board review recommended for consideration of second-line systemic therapy (e.g. Lomustine/Bevacizumab) or repeat surgical debulking."""
    },
    {
        "id": "sample_03_response",
        "title": "Sample 3: Durable Radiographic Complete Response",
        "scenario": "Complete Radiographic Remission",
        "mrn": "0044",
        "target_patient": "Priya Sharma (MRN: 0044)",
        "file_name": "03_complete_treatment_response.txt",
        "badge_color": "#0ea5e9",
        "expected_severity": "Mild / Reassuring",
        "expected_rano": "CR (Complete Response)",
        "summary": "Right parietal lobe, non-enhancing, normalized rCBV = 1.10, 0.0 mm midline shift, no mass effect.",
        "text": """CLINICAL RADIOLOGY & MULTI-PARAMETRIC MRI REPORT
PATIENT: Priya Sharma
MRN: 0044
DATE OF EXAMINATION: 2026-09-27
CLINICAL INDICATION: 41-year-old female with anaplastic oligodendroglioma (IDH-mutant, 1p/19q codeleted, WHO Grade 3) of the right parietal lobe. Completed 6 cycles of PCV chemotherapy. 6-month routine surveillance examination.

TECHNIQUE: Multi-sequence 3T brain MRI including 3D T1 MPRAGE, axial T2 fast spin echo, 3D FLAIR, DSC perfusion, and post-contrast volumetric 3D T1-weighted imaging.

FINDINGS:
BRAIN PARENCHYMA & TUMOR BED:
The previously demonstrated T2/FLAIR hyperintense lesion within the right parietal subcortical white matter demonstrates continued involution and marked decrease in extent. There is minimal residual non-enhancing FLAIR signal abnormality measuring approximately 3.20 cm3, compatible with chronic gliosis and post-treatment parenchymal remodeling.

CONTRAST ENHANCEMENT:
Following intravenous gadolinium administration, there is NO pathological nodular or rim contrast enhancement identified anywhere within the brain parenchyma. The surgical cavity and surrounding cortical architecture appear calm.

PERFUSION rCBV & MASS EFFECT:
Perfusion imaging reveals normalized blood flow through the region with an rCBV of 1.10. Ventricular size and sulcal prominence are symmetrical and appropriate for age. No mass effect, no midline shift (0.0 mm).

IMPRESSION:
1. Durable radiographic complete treatment response (RANO 2.0 Category: Complete Response / CR) with no evidence of tumor recurrence or residual high-grade disease.
2. Complete absence of abnormal enhancement and normalized microvascular perfusion (rCBV 1.10).
3. Recommend continued routine clinical surveillance with follow-up neuro-imaging in 6 months."""
    },
    {
        "id": "sample_04_stat",
        "title": "Sample 4: Acute Mass Effect & Herniation Alert",
        "scenario": "Critical Herniation Emergency",
        "mrn": "0042",
        "target_patient": "Emergency Consultation (MRN: 0042)",
        "file_name": "04_acute_mass_effect_herniation.txt",
        "badge_color": "#dc2626",
        "expected_severity": "Critical / STAT",
        "expected_rano": "Critical Urgent Triage",
        "summary": "Severe edema 24.5 cm³, 5.8 mm subfalcine midline shift, rCBV = 2.95, impending uncal herniation.",
        "text": """EMERGENCY NEURORADIOLOGY CONSULTATION REPORT
PATIENT: Emergency Triage / Acute Deterioration
MRN: 0042
DATE OF EXAMINATION: 2026-09-30
CLINICAL INDICATION: Acute neurological deterioration; refractory headaches, progressive lethargy, confusion, and new right upper extremity pronator drift following Cycle 3 temozolomide. Emergent evaluation for mass effect.

TECHNIQUE: Emergency high-resolution non-contrast and contrast-enhanced 3D volumetric Brain MRI.

FINDINGS:
MASS EFFECT & HERNIATION:
There is rapid expansion of extensive finger-like vasogenic edema involving the deep temporal and fronto-insular white matter, measuring 24.50 cm3. Severe mass effect is observed with marked compression and near-total effacement of the left lateral ventricle frontal and temporal horns. Subfalcine herniation is present with 5.8 mm rightward midline shift across the falx cerebri. Early effacement of the perimesencephalic cisterns noted.

LESION CHARACTERISTICS & PERFUSION:
Irregular peripheral rim enhancement measuring 4.2 x 3.6 cm. Central non-enhancing necrotic cavity. Perfusion rCBV is elevated at 2.95 at the anterior leading edge.

IMPRESSION:
1. CRITICAL ALERT: Severe intracranial mass effect with 5.8 mm subfalcine herniation and impending uncal herniation.
2. Extensive aggressive vasogenic edema with elevated perfusion (rCBV 2.95).
3. URGENT ACTION: Immediate neurosurgical consultation requested for evaluation of emergent surgical decompression. Initiate intravenous dexamethasone (10 mg IV bolus followed by 4 mg q6h) and hyperosmolar therapy with 23.4% hypertonic saline or mannitol as clinically indicated."""
    }
]


def render_report_verification_hub(
    patient: Dict[str, Any],
    db: Any,
    latest_scan: Dict[str, Any],
    molecular: Dict[str, Any],
    current_doc: Optional[Dict[str, Any]] = None
):
    """
    Renders the Clinical Report Upload & AI Verification Hub inside Clinician Workstation.
    """
    patient_id = patient["patient_id"]
    active_mrn = patient["mrn"]

    st.markdown("##### 📄 Clinical Radiology Report Ingestion & Verification Hub")
    st.markdown(
        "Upload clinical radiology reports (.txt, .md, .pdf) or select pre-configured clinical sample reports. "
        "Run the **Bio_ClinicalBERT NLP Engine** and **RANO 2.0 Triage Classifier** to extract diagnostic entities, "
        "verify perfusion rCBV thresholds, generate 6th-grade empathetic patient summaries, and commit validated reports to the Patient Digital Twin."
    )

    # =========================================================================
    # STEP 1: SAMPLE REPORTS LIBRARY & DOWNLOAD/LOAD SHELF
    # =========================================================================
    with st.expander("📚 Clinical Sample Reports Library (Click to Inspect, Test & Download)", expanded=True):
        st.markdown(
            "Select any standard clinical scenario below to **stage the report** into the uploader, "
            "or **download** the report text file to test drag-and-drop file ingestion."
        )

        cols = st.columns(4)
        for i, s in enumerate(SAMPLE_REPORTS_LIBRARY):
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="background: white; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 12px; min-height: 240px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <span style="background: {s['badge_color']}15; color: {s['badge_color']}; font-size: 0.72rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; text-transform: uppercase;">
                                {s['scenario']}
                            </span>
                            <h5 style="margin: 6px 0 4px 0; font-size: 0.95rem; color: #0f172a;">{s['title']}</h5>
                            <p style="margin: 0; font-size: 0.78rem; color: #64748b; line-height: 1.3;">
                                <strong>Patient:</strong> {s['target_patient']}<br>
                                <strong>Expected:</strong> {s['expected_severity']}
                            </p>
                            <p style="margin: 6px 0 0 0; font-size: 0.75rem; color: #334155; line-height: 1.25;">
                                {s['summary']}
                            </p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write("")
                col_btn_load, col_btn_dl = st.columns([1.2, 0.8])
                with col_btn_load:
                    if st.button("⚡ Stage Report", key=f"stage_btn_{s['id']}", use_container_width=True, type="primary"):
                        st.session_state.staged_report_text = s["text"]
                        st.session_state.staged_report_source = s["title"]
                        st.session_state.staged_report_matched_mrn = s["mrn"]
                        if s["mrn"] != st.session_state.get("active_patient_mrn"):
                            st.session_state.active_patient_mrn = s["mrn"]
                        st.success(f"Loaded: {s['title']}")
                        st.rerun()

                with col_btn_dl:
                    st.download_button(
                        label="⬇️ File",
                        data=s["text"],
                        file_name=s["file_name"],
                        mime="text/plain",
                        key=f"dl_btn_{s['id']}",
                        use_container_width=True
                    )

    # =========================================================================
    # STEP 2: INGESTION STAGING AREA (UPLOAD FILE OR EDIT TEXT)
    # =========================================================================
    st.markdown("##### 1. Stage Clinical Radiology Report for Active Patient")

    # If no staged text exists yet, default to active patient default narrative
    if "staged_report_text" not in st.session_state or not st.session_state.staged_report_text:
        st.session_state.staged_report_text = SAMPLE_REPORTS_LIBRARY[0]["text"]
        st.session_state.staged_report_source = "Pre-loaded Standard Follow-up"

    col_up, col_info = st.columns([1.8, 1.2])

    with col_up:
        uploaded_file = st.file_uploader(
            "📂 Upload External Clinical Report File (.txt, .md, .pdf, .json):",
            type=["txt", "md", "json", "pdf"],
            key="clinician_report_hub_file_uploader",
            help="Drop any clinical notes, consultation reports, or one of the downloaded sample files here."
        )

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.read()
                # Attempt standard utf-8 decoding
                file_text = raw_bytes.decode("utf-8", errors="replace")
                if file_text.strip():
                    st.session_state.staged_report_text = file_text
                    st.session_state.staged_report_source = f"Uploaded File: {uploaded_file.name}"
                    st.toast(f"✅ Staged {uploaded_file.name} successfully!")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")

    with col_info:
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 1.6rem;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Target Patient Record</span><br>
                <strong style="color: #0f172a; font-size: 1.05rem;">{patient['full_name']}</strong> (MRN: <code>{patient['mrn']}</code>)<br>
                <span style="font-size: 0.85rem; color: #475569;">{patient['diagnosis_type']} • {patient['diagnosis_location']}</span><br>
                <span style="font-size: 0.8rem; color: #0f766e; font-weight: 600;">Staged Source: {st.session_state.get('staged_report_source', 'Direct Input')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Live Staged Text Editor
    staged_text = st.text_area(
        "Clinical Radiology Narrative Findings & Impression (Live Editable):",
        value=st.session_state.staged_report_text,
        height=180,
        key="staged_report_narrative_editor"
    )
    # Synchronize edits back to session state
    st.session_state.staged_report_text = staged_text

    c_cnt1, c_cnt2, c_clear = st.columns([1, 1, 1])
    with c_cnt1:
        st.caption(f"📊 Character Count: `{len(staged_text)}` | Word Count: `{len(staged_text.split())}`")
    with c_clear:
        if st.button("🧹 Clear Staging Area", use_container_width=True):
            st.session_state.staged_report_text = ""
            st.session_state.staged_report_source = "Empty"
            st.session_state.current_verified_report_result = None
            st.rerun()

    st.write("")

    # =========================================================================
    # STEP 3: RUN BIO_CLINICALBERT & RANO 2.0 EXTRACTION ENGINE
    # =========================================================================
    st.markdown("##### 2. Execute AI Extraction & Diagnostic Triage Pipeline")

    btn_analyze = st.button(
        "🚀 Run Bio_ClinicalBERT Extraction & Verification Engine",
        type="primary",
        use_container_width=True
    )

    if btn_analyze:
        if not staged_text.strip():
            st.error("Please stage or upload a clinical report first.")
        else:
            with st.status("Executing Bio_ClinicalBERT NLP & RANO 2.0 Triage Pipeline...", expanded=True) as status:
                st.write("1. Initializing Bio_ClinicalBERT diagnostic semantic parser...")
                nlp_engine = ClinicalNLPEngine(offline=True)

                st.write("2. Extracting structured clinical entities (tumor location, margins, rCBV perfusion, edema, midline shift)...")
                nlp_res = nlp_engine.analyze_report(staged_text)

                st.write("3. Evaluating RANO 2.0 Pseudoprogression (PsP) Criteria and urgency triage...")
                triage_eng = RANO2TriageEngine()
                parsed_rcbv = nlp_res["parsed_entities"].get("rcbv", 1.50)
                shift_str = nlp_res["parsed_entities"].get("midline_shift", "0.0")
                try:
                    shift_val = float(shift_str.split()[0])
                except Exception:
                    shift_val = 0.0

                current_et_vol = latest_scan.get("et_vol_cm3", 4.10)
                triage_res = triage_eng.evaluate_response(
                    current_et_vol=current_et_vol,
                    prior_et_vol=latest_scan.get("et_vol_cm3", 4.10),
                    baseline_et_vol=8.20,
                    days_since_rt_completion=45,
                    rcbv=parsed_rcbv,
                    mgmt_methylated=molecular.get("mgmt_methylated", True),
                    midline_shift_mm=shift_val
                )

                st.write("4. Synthesizing 6th-grade empathetic patient translation and computing 11-dimensional normalized semantic vector...")
                status.update(label="✅ Clinical AI Extraction & Triage Analysis Complete!", state="complete")

                st.session_state.current_verified_report_result = {
                    "raw_text": staged_text,
                    "nlp_res": nlp_res,
                    "triage_res": triage_res,
                    "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

    # =========================================================================
    # STEP 4: VERIFICATION RESULTS DISPLAY & PHYSICIAN ATTESTATION CONSOLE
    # =========================================================================
    res = st.session_state.get("current_verified_report_result")
    if res:
        st.markdown("---")
        st.markdown("##### 3. AI Extraction Results & Diagnostic Verification Console")

        nlp_data = res["nlp_res"]
        triage_data = res["triage_res"]
        entities = nlp_data["parsed_entities"]
        severity_tag = nlp_data["severity_tag"]
        rano_cat = triage_data.get("rano_category", "PsP")
        psp_pct = triage_data.get("pseudoprogression_probability", 0.84) * 100.0

        # Top Diagnostic Banner
        border_col = "#ef4444" if severity_tag == "Critical" else ("#f59e0b" if severity_tag == "Moderate" else "#10b981")
        bg_col = "#fef2f2" if severity_tag == "Critical" else ("#fffbeb" if severity_tag == "Moderate" else "#f0fdf4")
        icon = "🚨" if severity_tag == "Critical" else ("⚠️" if severity_tag == "Moderate" else "✅")

        st.markdown(
            f"""
            <div style="background-color: {bg_col}; border: 1.5px solid {border_col}; border-radius: 10px; padding: 1.2rem; margin-bottom: 1.2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 700; color: {border_col}; text-transform: uppercase;">
                            {icon} AI Clinical Diagnostic Assessment ({res['analysis_timestamp']})
                        </span>
                        <h3 style="margin: 4px 0 0 0; color: #0f172a; font-size: 1.3rem;">
                            Clinical Severity: <span style="color: {border_col};">{severity_tag}</span> • RANO 2.0: <span style="color: #0f766e;">[{rano_cat}]</span>
                        </h3>
                        <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #334155;">
                            {triage_data.get('clinical_rationale', 'Diagnostic criteria evaluated.')}
                        </p>
                    </div>
                    <div style="text-align: right; background: white; padding: 10px 16px; border-radius: 8px; border: 1px solid #cbd5e1;">
                        <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b;">PsP Probability</span><br>
                        <strong style="font-size: 1.25rem; color: #0f766e;">{psp_pct:.1f}%</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tabular Diagnostic Entities
        col_ent, col_plain = st.columns([1.1, 0.9])

        with col_ent:
            st.markdown("###### 🔍 Extracted Clinical Diagnostic Entities (Bio_ClinicalBERT)")
            ent_rows = [
                {"Entity": "Tumor Anatomical Location", "Extracted Value": entities.get("tumor_location", "N/A"), "Significance": "Guides resection & target volume"},
                {"Entity": "Contrast Enhancement Margins", "Extracted Value": entities.get("contrast_enhancement", "N/A"), "Significance": "Quantifies active vascular disruption"},
                {"Entity": "Margin Invasiveness", "Extracted Value": entities.get("margins", "N/A"), "Significance": "Assesses tumor border infiltration"},
                {"Entity": "Peritumoral Edema Extent", "Extracted Value": entities.get("edema", "N/A"), "Significance": "Evaluates vasogenic fluid burden"},
                {"Entity": "Mass Effect & Ventricles", "Extracted Value": entities.get("mass_effect", "N/A"), "Significance": "Detects parenchyma crowding"},
                {"Entity": "Midline Shift", "Extracted Value": entities.get("midline_shift", "None (0 mm)"), "Significance": "Herniation risk marker (>3mm is urgent)"},
                {"Entity": "DSC Perfusion (rCBV)", "Extracted Value": f"{entities.get('rcbv', 1.50):.2f}", "Significance": "<1.75 favors necrosis/PsP; >2.0 favors recurrence"}
            ]
            st.dataframe(ent_rows, use_container_width=True)

        with col_plain:
            st.markdown("###### 🌟 Empathetic Plain-Language Translation (6th-Grade Reading Level)")
            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1rem; font-size: 0.92rem; line-height: 1.5; color: #0f172a;">
                    <p style="margin: 0;">
                        {nlp_data['plain_language_summary']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption("💡 Generated automatically for patient understanding and portal transparency.")

        # Semantic Embedding Vector Representation
        with st.expander("🧬 View 11-Dimensional L2-Normalized Semantic Feature Vector", expanded=False):
            st.caption("Normalized numerical embedding fed directly into the Multimodal Twin Cross-Attention Fusion Network:")
            st.code(str(nlp_data.get("embedding_11d", [])), language="json")

        # =========================================================================
        # STEP 5: PHYSICIAN ATTESTATION & DIGITAL TWIN COMMIT
        # =========================================================================
        st.markdown("##### 4. Attending Physician Verification & Electronic Sign-Off")

        doc_name = current_doc["full_name"] if current_doc else patient["oncologist_name"]
        doc_license = current_doc.get("license_no", "Institutional Board Certified") if current_doc else "MD Neuro-Oncology"

        with st.form("physician_report_verification_form"):
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                st.text_input("Verifying Attending Physician", value=doc_name, disabled=True)
            with c_v2:
                st.text_input("Physician Credentials / License", value=doc_license, disabled=True)

            physician_notes = st.text_area(
                "Attending Physician Clinical Notes & Action Plan:",
                value=f"Verified diagnostic NLP parsing. Clinical findings consistent with {rano_cat} category. Patient to continue scheduled therapy.",
                height=80
            )

            attest_box = st.checkbox(
                "☑️ I, the attending physician, verify that these diagnostic findings, NLP entity extractions, and triage assessments "
                "have been reviewed for accuracy and I authorize synchronizing this report with the Patient Digital Twin.",
                value=True
            )

            submit_commit = st.form_submit_button(
                "✅ Authorize & Commit Verified Report to Patient Digital Twin",
                type="primary",
                use_container_width=True
            )

            if submit_commit:
                if not attest_box:
                    st.error("Please check the verification attestation box to authorize this clinical report.")
                else:
                    new_rep_id = str(uuid.uuid4())
                    today_str = datetime.now().strftime("%Y-%m-%d")

                    # Generate verification hash
                    hash_input = f"{new_rep_id}-{patient_id}-{doc_name}-{today_str}"
                    verify_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16].upper()

                    # Insert report into database
                    db.insert_clinical_report({
                        "report_id": new_rep_id,
                        "patient_id": patient_id,
                        "scan_id": latest_scan.get("scan_id"),
                        "uploaded_by": f"Doctor (Verified: {doc_name})",
                        "report_date": today_str,
                        "raw_text": res["raw_text"],
                        "parsed_entities": entities,
                        "plain_language_summary": nlp_data["plain_language_summary"],
                        "severity_tag": severity_tag,
                        "verification_hash": verify_hash,
                        "physician_notes": physician_notes
                    })

                    # Update twin timeline
                    twin_alert = f"Verified {today_str}: [{rano_cat}] category. {physician_notes}"
                    db.update_twin_timeline({
                        "patient_id": patient_id,
                        "evaluation_date": today_str,
                        "progression_risk_score": 0.22 if rano_cat in ["PsP", "CR"] else (0.75 if rano_cat == "PD" else 0.40),
                        "triage_category": severity_tag,
                        "rano_category": rano_cat,
                        "pseudoprogression_risk": triage_data.get("pseudoprogression_probability", 0.84),
                        "is_pseudoprogression": triage_data.get("is_pseudoprogression", True),
                        "adherence_rate_pct": 92.5,
                        "alert_banner_text": twin_alert
                    })

                    st.session_state.last_committed_report_info = {
                        "report_id": new_rep_id,
                        "verify_hash": verify_hash,
                        "doc_name": doc_name,
                        "date": today_str,
                        "patient": patient["full_name"]
                    }
                    st.success(f"✅ Report successfully verified and committed for {patient['full_name']}!")
                    st.rerun()

    # If recently committed, display verification receipt banner
    receipt = st.session_state.get("last_committed_report_info")
    if receipt:
        st.markdown(
            f"""
            <div style="background-color: #ecfdf5; border: 1.5px solid #10b981; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #065f46; font-size: 1.05rem;">🎉 Electronic Verification Certificate Issued</strong><br>
                        <span style="font-size: 0.85rem; color: #047857;">
                            Report ID: <code>{receipt['report_id']}</code> • Digital Signature Hash: <code>{receipt['verify_hash']}</code><br>
                            Attending: <strong>{receipt['doc_name']}</strong> • Target: <strong>{receipt['patient']}</strong> • Date: {receipt['date']}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================================================================
    # STEP 6: HISTORICAL VERIFIED REPORTS FOR ACTIVE PATIENT
    # =========================================================================
    st.markdown("---")
    st.markdown(f"##### 📋 Verified Consultation & Radiology Reports on File ({patient['full_name']})")
    all_reports = db.get_patient_reports(patient_id)

    if all_reports:
        for idx, rep in enumerate(all_reports):
            rep_date = rep.get("report_date", "Recent")
            rep_uploader = rep.get("uploaded_by", "Clinician")
            rep_sev = rep.get("severity_tag", "Moderate")
            sev_col = "#ef4444" if rep_sev == "Critical" else ("#f59e0b" if rep_sev == "Moderate" else "#10b981")

            with st.expander(f"📄 Report #{idx+1} • {rep_date} • Uploaded by: {rep_uploader} • Severity: [{rep_sev}]", expanded=(idx == 0)):
                st.markdown(f"**Verified By / Source**: `{rep_uploader}` | **Clinical Severity**: <span style='color:{sev_col}; font-weight:700;'>{rep_sev}</span>", unsafe_allow_html=True)
                st.markdown("**Patient-Friendly Summary:**")
                st.info(rep.get("plain_language_summary", "No summary available."))
                st.markdown("**Original Clinical Findings:**")
                st.code(rep.get("raw_text", ""), language="text")
    else:
        st.info("No clinical reports currently recorded for this patient.")
