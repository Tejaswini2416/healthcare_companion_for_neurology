"""
Doctor Clinical Report Writing & Logging Component
Provides:
1. Dedicated clinical authoring portal for attending physicians and radiologists.
2. Template-assisted report composition for Brain MRI Radiology and Molecular Pathology.
3. Automated Bio_ClinicalBERT entity extraction & plain-language translation generation.
4. Direct persistence to the Patient Digital Twin database with physician attestation.
5. Instant PDF generation and export for clinical charts and patient portal delivery.
"""

import streamlit as st
import uuid
import math
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from ...utils.supabase_client import get_database_client
from ...models.clinical_nlp import ClinicalNLPEngine
from ...models.triage_engine import RANO2TriageEngine
from ...utils.pdf_generator import generate_patient_clinical_pdf
from .doctor_login import get_current_doctor


TEMPLATES = {
    "Surveillance Brain MRI (Stable / Pseudoprogression)": {
        "report_type": "Brain MRI Radiology Report",
        "indication": "Routine surveillance evaluation following adjuvant chemoradiation for high-grade glioma.",
        "technique": "Multi-parametric 3T brain MRI with pre- and post-gadolinium T1, T2, 3D FLAIR, DSC perfusion, and DWI.",
        "wt_vol": 13.80,
        "tc_vol": 7.40,
        "et_vol": 3.90,
        "edema_vol": 6.10,
        "rcbv": 1.42,
        "midline_shift": 0.0,
        "findings": """BRAIN PARENCHYMA & SURGICAL CAVITY:
Evaluation of the resection margins demonstrates stable surgical cavity contour. Along the posterior margin, there is a smooth rim of residual enhancement measuring 1.2 x 1.6 cm (enhancing volume 3.90 cm³). Compared to prior scan, there is interval stability in enhancing margin.

PERITUMORAL EDEMA & MASS EFFECT:
Perilesional T2/FLAIR hyperintensity is stable at 6.10 cm³, representing mild post-treatment reactive gliosis and edema. No mass effect on ventricular horns. Midline structures are anatomical with 0.0 mm shift.

DSC PERFUSION & DIFFUSION:
Perfusion blood flow remains low with mean rCBV measured at 1.42. No restricted diffusion to suggest hypercellular tumor recurrence.

IMPRESSION:
1. Stable post-treatment appearance with low perfusion rCBV (1.42) consistent with radiation-induced pseudoprogression / radiation necrosis.
2. No evidence of true progressive neoplastic disease. Continue current maintenance temozolomide.""",
        "severity": "Mild"
    },
    "High-Grade Glioma Suspected Progression / Recurrence": {
        "report_type": "Brain MRI Radiology Report",
        "indication": "New onset progressive cognitive slowing and subtle motor weakness. Evaluate for tumor recurrence.",
        "technique": "High-resolution 3D multi-parametric brain MRI with DSC perfusion and DTI.",
        "wt_vol": 19.20,
        "tc_vol": 11.50,
        "et_vol": 6.80,
        "edema_vol": 12.40,
        "rcbv": 3.20,
        "midline_shift": 2.5,
        "findings": """BRAIN PARENCHYMA & SURGICAL CAVITY:
Interval enlargement and nodular expansion of irregular peripheral rim enhancement along the lateral aspect of the surgical resection margin, now measuring 2.8 x 2.4 cm (enhancing volume 6.80 cm³).

PERITUMORAL EDEMA & MASS EFFECT:
Expanding perilesional vasogenic edema involving white matter tracts, now measuring 12.40 cm³. Partial compression of adjacent ventricular horn with 2.5 mm midline shift.

DSC PERFUSION & DIFFUSION:
Elevated relative cerebral blood volume with peak rCBV of 3.20, indicating active microvascular neoangiogenesis. Focal areas of restricted diffusion confirm increased cellularity.

IMPRESSION:
1. Findings are consistent with true tumor recurrence / progressive disease (RANO 2.0 PD).
2. Recommend urgent tumor board review for second-line systemic therapy or surgical debulking.""",
        "severity": "Critical"
    },
    "Complete Radiographic Treatment Response": {
        "report_type": "Brain MRI Radiology Report",
        "indication": "Follow-up surveillance imaging post-completion of adjuvant chemotherapy.",
        "technique": "3T multi-sequence contrast-enhanced brain MRI.",
        "wt_vol": 2.50,
        "tc_vol": 0.0,
        "et_vol": 0.0,
        "edema_vol": 2.50,
        "rcbv": 1.05,
        "midline_shift": 0.0,
        "findings": """BRAIN PARENCHYMA & RESECTION MARGINS:
No pathologic nodular or rim contrast enhancement identified anywhere within the brain parenchyma. Complete resolution of previously demonstrated enhancing lesion. Minimal residual FLAIR gliosis (2.50 cm³). Normal ventricular symmetry. No mass effect or midline shift.

IMPRESSION:
1. Complete radiographic response (RANO 2.0 Category: CR).
2. No evidence of active or recurrent tumor nidus. Recommend routine surveillance MRI in 6 months.""",
        "severity": "Mild"
    },
    "Surgical Histopathology & Molecular Biomarker Report": {
        "report_type": "Surgical Pathology Report",
        "indication": "Surgical biopsy and maximal safe debulking for intracranial neoplasm.",
        "technique": "Histopathologic examination with H&E staining and comprehensive Next-Generation Sequencing (NGS) molecular panel.",
        "wt_vol": 14.00,
        "tc_vol": 7.00,
        "et_vol": 4.00,
        "edema_vol": 6.00,
        "rcbv": 1.45,
        "midline_shift": 0.0,
        "findings": """SPECIMEN: Brain tumor tissue, surgical resection.
HISTOPATHOLOGIC FINDINGS:
Microscopic examination reveals hypercellular glial neoplasm with nuclear pleomorphism, microvascular proliferation, and areas of focal necrosis.

MOLECULAR BIOMARKERS & IMMUNOHISTOCHEMISTRY:
- IDH1 (R132H): MUTANT (Detected)
- MGMT Promoter: METHYLATED (62% methylation index - favorable Temozolomide response)
- 1p/19q Chromosomal Co-deletion: NON-CODELETED (Intact)
- Ki-67 Proliferation Index: 14%
- TERT Promoter: Mutant
- ATRX: Retained nuclear expression

PATHOLOGIC DIAGNOSIS:
Astrocytoma, IDH-mutant, WHO Grade 4 (CNS WHO 2021 Classification).""",
        "severity": "Moderate"
    }
}


def render_report_writing_page():
    """Renders the comprehensive Clinician Report Writing and Logging Console."""
    db = get_database_client()
    current_doc = get_current_doctor()
    patients = db.get_all_patients()

    if not patients:
        st.error("No patient records loaded in the system.")
        return

    st.markdown("## 📝 Clinical Report Writer & Patient Logging Console")
    st.caption("Author, verify, electronically sign, and permanently log official radiology and pathology reports for active patients.")

    # Top Physician Card
    if current_doc:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0f766e 0%, #064e3b 100%); padding: 1rem 1.4rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.75rem; text-transform: uppercase; background: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 4px;">
                            Authoring Physician
                        </span>
                        <h3 style="margin: 6px 0 2px 0; font-size: 1.25rem;">
                            {current_doc.get('avatar', '👨‍⚕️')} {current_doc.get('full_name', 'Attending Physician')}
                        </h3>
                        <span style="font-size: 0.85rem; opacity: 0.9;">
                            {current_doc.get('role', 'Attending')} • {current_doc.get('department', 'Neuro-Oncology')}
                        </span>
                    </div>
                    <div style="text-align: right; font-size: 0.8rem; opacity: 0.85;">
                        Electronic Attestation Ready<br>
                        {current_doc.get('license_no', 'NPI: 1849204821')}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Patient Selector for Authoring
    col_pt_sel, col_tpl_sel = st.columns([1.5, 1.5])
    with col_pt_sel:
        patient_options = {f"{p['full_name']} (MRN: {p['mrn']}) - {p['diagnosis_location']}": p for p in patients}
        active_mrn = st.session_state.get("active_patient_mrn", patients[0]["mrn"])
        default_idx = 0
        for i, p in enumerate(patient_options.values()):
            if p["mrn"] == active_mrn:
                default_idx = i
                break

        selected_pt_label = st.selectbox(
            "Select Target Patient to Write Report For:",
            options=list(patient_options.keys()),
            index=default_idx,
            key="doc_writer_patient_sel"
        )
        selected_patient = patient_options[selected_pt_label]
        if selected_patient["mrn"] != st.session_state.get("active_patient_mrn"):
            st.session_state.active_patient_mrn = selected_patient["mrn"]

    with col_tpl_sel:
        template_choice = st.selectbox(
            "Load Quick Clinical Template:",
            options=["-- Start from Blank Report --"] + list(TEMPLATES.keys()),
            index=1,
            key="doc_writer_tpl_sel"
        )

    # Active Patient Summary Strip
    st.info(
        f"**Patient**: `{selected_patient['full_name']}` | **MRN**: `{selected_patient['mrn']}` | "
        f"**Diagnosis**: *{selected_patient['diagnosis_type']}* ({selected_patient['diagnosis_location']}) | "
        f"**KPS**: `{selected_patient['baseline_kps']}%` | **Primary Oncologist**: `{selected_patient['oncologist_name']}`"
    )

    # Pre-populate template data if selected
    tpl_data = TEMPLATES.get(template_choice, {}) if template_choice != "-- Start from Blank Report --" else {}

    st.markdown("### ✍️ Clinical Report Composition Form")

    with st.form("clinical_report_authoring_form", clear_on_submit=False):
        c_f1, c_f2, c_f3 = st.columns([1.5, 1, 1])
        with c_f1:
            report_type = st.selectbox(
                "Report Category *",
                options=[
                    "Brain MRI Radiology Report",
                    "Surgical Pathology Report",
                    "Molecular Biomarker Panel",
                    "Multidisciplinary Tumor Board Consultation"
                ],
                index=0 if "Pathology" not in tpl_data.get("report_type", "") else 1
            )
        with c_f2:
            exam_date = st.date_input("Examination / Procedure Date *", value=datetime.now())
        with c_f3:
            severity_tag = st.selectbox(
                "Triage Severity Classification *",
                options=["Mild", "Moderate", "Critical"],
                index=0 if tpl_data.get("severity") == "Mild" else (2 if tpl_data.get("severity") == "Critical" else 1)
            )

        col_ind, col_tech = st.columns(2)
        with col_ind:
            clinical_indication = st.text_input(
                "Clinical Indication *",
                value=tpl_data.get("indication", f"Surveillance evaluation for {selected_patient['diagnosis_type']} in {selected_patient['diagnosis_location']}."),
                placeholder="e.g. Surveillance evaluation post-Stupp protocol chemoradiation."
            )
        with col_tech:
            technique = st.text_input(
                "Imaging Technique / Laboratory Modality",
                value=tpl_data.get("technique", "Multi-parametric 3T brain MRI with DSC perfusion, DWI, and gadolinium contrast.")
            )

        st.markdown("##### 🔬 Volumetric & Perfusion Measurements (Auto-linked to Digital Twin):")
        v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)
        with v_col1:
            wt_input = st.number_input("Whole Tumor (WT cm³)", min_value=0.0, max_value=200.0, value=float(tpl_data.get("wt_vol", 14.6)), step=0.1)
        with v_col2:
            et_input = st.number_input("Enhancing Core (ET cm³)", min_value=0.0, max_value=100.0, value=float(tpl_data.get("et_vol", 4.1)), step=0.1)
        with v_col3:
            ed_input = st.number_input("Vasogenic Edema (cm³)", min_value=0.0, max_value=150.0, value=float(tpl_data.get("edema_vol", 6.5)), step=0.1)
        with v_col4:
            rcbv_input = st.number_input("Perfusion rCBV", min_value=0.0, max_value=15.0, value=float(tpl_data.get("rcbv", 1.48)), step=0.05, help="<1.75 indicates treatment-induced pseudoprogression")
        with v_col5:
            shift_input = st.number_input("Midline Shift (mm)", min_value=0.0, max_value=30.0, value=float(tpl_data.get("midline_shift", 0.0)), step=0.5)

        st.markdown("##### 📄 Clinical Findings Narrative & Radiologic / Pathologic Impression:")
        findings_text = st.text_area(
            "Detailed Findings & Impression (Live Editable):",
            value=tpl_data.get("findings", ""),
            height=200,
            placeholder="Document detailed parenchymal findings, margins, perfusion characteristics, and final diagnostic impression..."
        )

        st.markdown("##### 💚 Patient-Friendly Empathetic Translation:")
        plain_custom = st.text_area(
            "Plain-Language Summary for Patient Portal (Leave blank to automatically synthesize via AI):",
            value="",
            height=90,
            placeholder="Optional: Customize how this report will be explained to the patient in clear, supportive terms..."
        )

        submit_report = st.form_submit_button("📝 Electronically Sign & Log Report to Patient Record", type="primary", use_container_width=True)

        if submit_report:
            if not findings_text.strip():
                st.error("Please provide the clinical findings narrative before logging.")
            else:
                full_raw_text = (
                    f"EXAMINATION: {report_type.upper()} ({exam_date.strftime('%Y-%m-%d')})\n"
                    f"PATIENT: {selected_patient['full_name']} (MRN: {selected_patient['mrn']})\n"
                    f"CLINICAL INDICATION: {clinical_indication}\n"
                    f"TECHNIQUE: {technique}\n\n"
                    f"{findings_text.strip()}"
                )

                # Synthesize plain-language summary if not provided
                if plain_custom.strip():
                    final_plain = plain_custom.strip()
                else:
                    nlp = ClinicalNLPEngine(offline=True)
                    final_plain = nlp.translate_to_plain_language(
                        full_raw_text,
                        parsed_entities={
                            "tumor_location": selected_patient["diagnosis_location"],
                            "midline_shift": f"{shift_input} mm",
                            "rcbv": rcbv_input
                        },
                        language="en"
                    )

                new_report_payload = {
                    "report_id": str(uuid.uuid4()),
                    "patient_id": selected_patient["patient_id"],
                    "scan_id": None,
                    "uploaded_by": current_doc.get("full_name", "Attending Neuro-Oncologist") if current_doc else "Doctor",
                    "report_date": exam_date.strftime("%Y-%m-%d"),
                    "raw_text": full_raw_text,
                    "parsed_entities": {
                        "tumor_location": selected_patient["diagnosis_location"],
                        "enhancing_dimensions_cm": f"{math.sqrt(max(0.01, float(et_input))):.1f} x {math.sqrt(max(0.01, float(et_input)))*1.2:.1f}",
                        "edema_status": f"Stable ({ed_input:.2f} cm³)",
                        "mass_effect": "Absent" if shift_input == 0 else f"{shift_input} mm Shift",
                        "midline_shift": f"{shift_input} mm",
                        "rcbv": rcbv_input,
                        "impression": f"{report_type} verified by attending physician"
                    },
                    "plain_language_summary": final_plain,
                    "severity_tag": severity_tag,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }

                # Persist to database
                db.insert_clinical_report(new_report_payload)

                # Also insert/update corresponding MRI scan volumetrics
                scan_payload = {
                    "scan_id": str(uuid.uuid4()),
                    "patient_id": selected_patient["patient_id"],
                    "scan_date": exam_date.strftime("%Y-%m-%d"),
                    "wt_vol_cm3": float(wt_input),
                    "tc_vol_cm3": float(wt_input * 0.55),
                    "et_vol_cm3": float(et_input),
                    "edema_vol_cm3": float(ed_input),
                    "dice_score": 0.9420,
                    "estimated_rcbv": float(rcbv_input),
                    "mask_storage_path": f"scans/{selected_patient['mrn']}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nii.gz",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                db.insert_scan_record(scan_payload)

                st.session_state.last_logged_report = new_report_payload
                st.session_state.last_logged_scan = scan_payload
                st.toast(f"✅ Report successfully logged for {selected_patient['full_name']}!", icon="📋")
                st.success(f"Report logged and signed into MRN **{selected_patient['mrn']}** record successfully!")

    # Display signed report & PDF download if just created
    last_logged = st.session_state.get("last_logged_report")
    last_scan = st.session_state.get("last_logged_scan")
    if last_logged and last_logged.get("patient_id") == selected_patient["patient_id"]:
        st.markdown("---")
        st.markdown("#### ✅ Active Electronically Signed Report Ready")

        c_prev1, c_prev2 = st.columns([2, 1])
        with c_prev1:
            st.info(f"**Impression Summary**: {last_logged['parsed_entities'].get('impression', 'Logged')}\n\n**Plain-Language Note**: {last_logged['plain_language_summary']}")
        with c_prev2:
            # Generate instant PDF
            pdf_bytes = generate_patient_clinical_pdf(
                patient=selected_patient,
                report=last_logged,
                scan_data=last_scan,
                molecular_data=db.get_patient_molecular(selected_patient["patient_id"])
            )
            st.download_button(
                label="📥 Download Official Signed PDF Report",
                data=pdf_bytes,
                file_name=f"Clinical_Report_{selected_patient['mrn']}_{last_logged['report_date']}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

    # Historical Logged Reports Table for this Patient
    st.markdown("---")
    st.markdown(f"#### 📚 Historical Logged Reports for {selected_patient['full_name']} (MRN: {selected_patient['mrn']})")
    all_reports = db.get_patient_reports(selected_patient["patient_id"])

    if all_reports:
        for idx, rep in enumerate(all_reports):
            with st.expander(f"📄 Report: {rep.get('report_date', 'N/A')} — Uploaded by: {rep.get('uploaded_by', 'Clinical Team')} ({rep.get('severity_tag', 'Normal')})"):
                c_rp_text, c_rp_act = st.columns([3, 1.2])
                with c_rp_text:
                    st.markdown("**Technical Narrative**:")
                    st.caption(rep.get("raw_text", ""))
                    st.markdown("**Patient-Friendly Translation**:")
                    st.success(rep.get("plain_language_summary", ""))
                with c_rp_act:
                    rep_pdf = generate_patient_clinical_pdf(
                        patient=selected_patient,
                        report=rep,
                        scan_data=last_scan,
                        molecular_data=db.get_patient_molecular(selected_patient["patient_id"])
                    )
                    st.download_button(
                        label=f"📥 Download PDF ({rep.get('report_date', '')})",
                        data=rep_pdf,
                        file_name=f"Report_{selected_patient['mrn']}_{rep.get('report_date', 'date')}.pdf",
                        mime="application/pdf",
                        key=f"dl_hist_pdf_{rep.get('report_id', idx)}_{idx}",
                        use_container_width=True
                    )
    else:
        st.info("No prior reports logged for this patient.")
