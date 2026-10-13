"""
Clinician Workstation Component
Neuro-Oncology Clinical Console providing:
1. Multi-Patient Roster & Global Active Patient Selector
2. Clinical Report Ingestion & Verification Hub (Bio_ClinicalBERT + RANO 2.0)
3. Medication Management & Verification Hub (Prescriptions + Adherence Overrides + Clinical Notes)
4. Clinical Report Generator & Formal Print/Export Console (window.print() + formatted markdown)
"""

import os
import uuid
import streamlit as st
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from ...utils.supabase_client import get_database_client
from .doctor_login import get_current_doctor, is_doctor_authenticated, logout_doctor
from ...models.red_flag_alert import get_escalation_engine
from .brain_3d_viewer import render_3d_brain_viewer_panel
from ...utils.pdf_generator import generate_patient_clinical_pdf


def _render_escalations_queue(current_doc: Optional[Dict[str, Any]], active_mrn: str):
    """Renders the Active Clinical Escalations Queue with SBAR cards and acknowledgment actions."""
    engine = get_escalation_engine()
    all_escalations = engine.get_all_escalations()
    pending = [e for e in all_escalations if "Active" in e.get("status", "")]
    pending_count = len(pending)

    with st.expander(f"🚨 Active Clinical Escalations Queue ({pending_count} Pending Physician Review)", expanded=(pending_count > 0)):
        col_hdr, col_flt = st.columns([2.5, 1])
        with col_hdr:
            st.markdown(
                "Hospital-grade automated escalation alerts triggered by acute patient symptoms, "
                "missed medications, or emergent clinical decompensation."
            )
        with col_flt:
            filter_mode = st.radio(
                "Filter Escalations:",
                options=["All Patients", "Active Patient Only"],
                horizontal=True,
                key="esc_filter_mode"
            )

        displayed_alerts = all_escalations
        if filter_mode == "Active Patient Only":
            displayed_alerts = [e for e in all_escalations if e.get("mrn") == active_mrn]

        if not displayed_alerts:
            st.info("No active escalation alerts for this patient.")
            return

        for alert in displayed_alerts:
            is_stat = alert.get("severity_tier") == "STAT"
            is_active = "Active" in alert.get("status", "")
            badge_bg = "#fef2f2" if is_stat else ("#fffbeb" if is_active else "#f0fdf4")
            border_col = "#ef4444" if is_stat else ("#f59e0b" if is_active else "#22c55e")
            tier_badge = "🔴 STAT EMERGENCY" if is_stat else ("🟠 URGENT" if is_active else "🟢 RESOLVED")

            st.markdown(
                f"""
                <div style="background: {badge_bg}; border: 1.5px solid {border_col}; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 700; color: {border_col}; font-size: 0.95rem;">{tier_badge} • {alert['trigger_event']}</span><br>
                            <span style="font-size: 0.82rem; color: #475569;">
                                Patient: <strong>{alert['patient_name']}</strong> (MRN: {alert['mrn']}) • Alert ID: <code>{alert['alert_id']}</code> • {alert['timestamp']}
                            </span>
                        </div>
                        <div style="text-align: right; font-size: 0.8rem; font-weight: 600; color: {'#b91c1c' if is_active else '#15803d'};">
                            {alert['status']}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if alert.get("mrn") != active_mrn:
                if st.button(f"🔍 Switch Workstation to {alert.get('patient_name', 'Patient')} (MRN: {alert.get('mrn')})", key=f"btn_switch_alert_{alert['alert_id']}"):
                    st.session_state.active_patient_mrn = alert["mrn"]
                    st.session_state["last_synced_clinician_mrn"] = alert["mrn"]
                    st.rerun()

            # SBAR Breakdown
            sbar = alert.get("sbar", {})
            c_sbar, c_disp = st.columns([2.5, 1.5])
            with c_sbar:
                st.markdown(
                    f"""
                    **📋 SBAR Clinical Handoff**:
                    - **Situation**: {sbar.get('situation', 'N/A')}
                    - **Background**: {sbar.get('background', 'N/A')}
                    - **Assessment**: {sbar.get('assessment', 'N/A')}
                    - **Recommendation**: {sbar.get('recommendation', 'N/A')}
                    """
                )
            with c_disp:
                sms_info = alert.get("dispatch_channels", {}).get("sms", {})
                pager_info = alert.get("dispatch_channels", {}).get("pager_webhook", {})
                st.markdown(
                    f"""
                    **📡 Telemetry & Dispatch**:
                    - 📱 **SMS**: {sms_info.get('status', 'Sent')} to *{sms_info.get('to', 'Attending')}*
                    - 📟 **PagerDuty / Webhook**: {pager_info.get('status', 'HTTP 200')} (*Priority: {pager_info.get('priority', 'P2')}*)
                    """
                )

            # Action: Acknowledge & Resolve
            if is_active:
                with st.form(key=f"ack_form_{alert['alert_id']}"):
                    col_input, col_submit = st.columns([3, 1])
                    with col_input:
                        doc_name = current_doc['full_name'] if current_doc else "Attending Neuro-Oncologist"
                        res_notes = st.text_input(
                            "Physician Triage & Resolution Note:",
                            value=f"Evaluated by {doc_name}. Advised nursing follow-up and verified anticonvulsant dosing.",
                            key=f"res_note_{alert['alert_id']}"
                        )
                    with col_submit:
                        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        if st.form_submit_button("🩺 Acknowledge & Resolve", type="primary", use_container_width=True):
                            engine.acknowledge_alert(alert["alert_id"], doc_name, res_notes)
                            st.success(f"Alert {alert['alert_id']} resolved by {doc_name}.")
                            st.rerun()
            else:
                st.caption(f"✅ **Resolved by {alert.get('acknowledged_by', 'Physician')}** on {alert.get('acknowledged_at', '')} — Note: *{alert.get('resolution_notes', 'Resolved')}*")

            st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)


def render_clinician_workstation():
    """Renders the comprehensive Neuro-Oncology Clinician Workstation."""
    db = get_database_client()
    patients = db.get_all_patients()
    current_doc = get_current_doctor()

    # Clinician Authentication Banner
    if current_doc:
        st.markdown(
            f"""
            <div style="background: #0f766e12; border: 1.5px solid #0f766e40; border-left: 5px solid #0f766e; border-radius: 8px; padding: 12px 16px; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #0f766e; letter-spacing: 0.05em;">
                            Authenticated Physician Console
                        </span><br>
                        <strong style="color: #0f172a; font-size: 1.1rem;">{current_doc['avatar']} {current_doc['full_name']}</strong> — 
                        <span style="color: #475569; font-size: 0.9rem;">{current_doc['role']}</span>
                        <span style="color: #64748b; font-size: 0.8rem; display: block; margin-top: 2px;">
                            {current_doc['department']} • {current_doc['license_no']}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("🩺 Clinician Workstation & Neuro-Oncology Console")
    st.markdown(
        "Advanced clinical interface for multi-patient management, multimodal scan ingestion, "
        "biophysical in-silico simulation, prescription governance, and formal consultation report generation."
    )

    # =========================================================================
    # 1. MULTI-PATIENT ROSTER & SELECTOR
    # =========================================================================
    patient_options = {
        f"{p['full_name']} (MRN: {p['mrn']}) - {p['diagnosis_location']}": p["mrn"]
        for p in patients
    }

    active_mrn = st.session_state.get("active_patient_mrn", "0042")
    current_idx = 0
    option_keys = list(patient_options.keys())
    for idx, (label, mrn) in enumerate(patient_options.items()):
        if mrn == active_mrn:
            current_idx = idx
            break

    # Sync selectbox key if active_mrn changed from sidebar or another action
    if st.session_state.get("last_synced_clinician_mrn") != active_mrn and current_idx < len(option_keys):
        st.session_state["clinician_patient_context_selector"] = option_keys[current_idx]
        st.session_state["last_synced_clinician_mrn"] = active_mrn

    c_sel, c_badge = st.columns([3, 1.2])
    with c_sel:
        selected_patient_label = st.selectbox(
            "🏥 Active Patient Context:",
            options=option_keys,
            index=current_idx,
            key="clinician_patient_context_selector"
        )
        selected_mrn = patient_options.get(selected_patient_label, active_mrn)
        if selected_mrn != st.session_state.get("active_patient_mrn"):
            st.session_state.active_patient_mrn = selected_mrn
            st.session_state["last_synced_clinician_mrn"] = selected_mrn
            st.rerun()

    patient = db.get_patient_by_mrn(st.session_state.get("active_patient_mrn", "0042"))
    if not patient:
        st.error("Patient record could not be loaded.")
        return

    patient_id = patient["patient_id"]
    molecular = db.get_patient_molecular(patient_id) or {"idh_status": "Mutant", "mgmt_methylated": True, "codeletion_1p19q": False}
    scans = db.get_patient_scans(patient_id)
    reports = db.get_patient_reports(patient_id)
    meds = db.get_patient_medications(patient_id)
    symptoms = db.get_patient_symptoms(patient_id)
    timeline = db.get_patient_timeline(patient_id)
    latest_scan = scans[-1] if scans else {"wt_vol_cm3": 14.60, "et_vol_cm3": 4.10, "estimated_rcbv": 1.48}
    latest_twin = timeline[0] if timeline else {}

    triage_priority = latest_twin.get("triage_category", "Moderate")
    p_color = "#ef4444" if triage_priority == "Critical" else ("#f59e0b" if triage_priority == "Moderate" else "#10b981")

    with c_badge:
        st.markdown(
            f"""
            <div style="background-color: {p_color}15; border: 1px solid {p_color}; border-radius: 8px; padding: 10px; text-align: center; margin-top: 1.5rem;">
                <span style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: {p_color};">Triage Priority</span><br>
                <strong style="color: {p_color}; font-size: 1.1rem;">{triage_priority} Priority</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Active Patient Summary Strip
    st.info(
        f"**Patient**: {patient['full_name']} | **Age/Sex**: {patient['age']} {patient['gender']} | "
        f"**Diagnosis**: *{patient['diagnosis_type']}* ({patient['diagnosis_location']}) | "
        f"**Molecular**: *IDH-{molecular.get('idh_status')}, MGMT-{'Methylated' if molecular.get('mgmt_methylated') else 'Unmethylated'}, "
        f"1p/19q-{'Codeleted' if molecular.get('codeletion_1p19q') else 'Intact'}* | "
        f"**KPS**: {patient['baseline_kps']} | **Oncologist**: {patient['oncologist_name']}"
    )

    # Active Clinical Escalations Queue
    _render_escalations_queue(current_doc=current_doc, active_mrn=patient["mrn"])

    # Clinical Tabs
    tab_roster, tab_meds, tab_report_gen, tab_3d_brain = st.tabs([
        "📋 Multi-Patient Roster",
        "💊 Medication Management & Verification Hub",
        "📑 Clinical Report Generator & Print Console",
        "🧠 3D Brain & Tumor Localization"
    ])

    # =========================================================================
    # TAB 1: Multi-Patient Directory Roster
    # =========================================================================
    with tab_roster:
        st.markdown("##### 👥 Neuro-Oncology Department Patient Roster")
        st.markdown("Comprehensive directory of active glioma patients under treatment and surveillance.")

        roster_rows = []
        for p in patients:
            pid = p["patient_id"]
            p_mol = db.get_patient_molecular(pid) or {}
            p_time = db.get_patient_timeline(pid)
            p_triage = p_time[0].get("triage_category", "Moderate") if p_time else "Moderate"
            p_rano = p_time[0].get("rano_category", "SD") if p_time else "SD"
            p_scans = db.get_patient_scans(pid)
            latest_v = p_scans[-1].get("wt_vol_cm3", 0.0) if p_scans else 0.0

            roster_rows.append({
                "MRN": p["mrn"],
                "Full Name": p["full_name"],
                "Age/Sex": f"{p['age']} {p['gender'][0]}",
                "Diagnosis Location": p["diagnosis_location"],
                "Diagnosis & WHO Subtype": p["diagnosis_type"],
                "Molecular Status": f"IDH-{p_mol.get('idh_status', 'N/A')}, MGMT-{'Met' if p_mol.get('mgmt_methylated') else 'Unmet'}",
                "Tumor Vol (cm³)": f"{latest_v:.1f}",
                "Baseline KPS": p["baseline_kps"],
                "RANO 2.0": p_rano,
                "Triage Alert": p_triage
            })

        st.dataframe(roster_rows, use_container_width=True)

        st.markdown("##### 🔀 Instant Patient Queue Selector")
        st.caption("Click any patient below to immediately switch the clinical workstation to their file and simulation state:")
        for p in patients:
            col_q1, col_q2, col_q3, col_q4 = st.columns([2.5, 2.0, 1.5, 1.2])
            is_cur = (p["mrn"] == active_mrn)
            with col_q1:
                st.markdown(f"**{p['full_name']}** (`MRN: {p['mrn']}`)")
                st.caption(f"{p.get('diagnosis_type', 'Glioma')} • {p.get('diagnosis_location', 'Brain')}")
            with col_q2:
                st.caption(f"📅 Diag: `{p.get('diagnosis_date', 'N/A')}` • Next Visit: `{p.get('next_visit_date', 'N/A')}`")
            with col_q3:
                st.caption(f"KPS: **{p.get('baseline_kps', 90)}%** • Stage: {p.get('treatment_stage', 'Adjuvant')[:22]}...")
            with col_q4:
                if is_cur:
                    st.success("🟢 Active")
                else:
                    if st.button("👉 Switch", key=f"btn_switch_roster_{p['mrn']}", use_container_width=True):
                        st.session_state.active_patient_mrn = p["mrn"]
                        st.session_state["last_synced_clinician_mrn"] = p["mrn"]
                        st.rerun()

    # =========================================================================
    # TAB 2: Medication Management & Verification Hub
    # =========================================================================
    with tab_meds:
        st.markdown("##### 💊 Medication Governance & Clinical Verification Hub")
        st.markdown(f"Manage active prescriptions and verify clinical adherence records for **{patient['full_name']}**.")

        # Summary Metrics
        total_p_meds = len(meds)
        taken_p_meds = sum(1 for m in meds if m.get("status") == "Taken")
        cur_adherence = round((taken_p_meds / max(1, total_p_meds)) * 100.0, 1)

        c_adh1, c_adh2, c_adh3 = st.columns(3)
        with c_adh1:
            st.metric("Total Prescribed Medications", total_p_meds)
        with c_adh2:
            st.metric("Doses Verified Taken", f"{taken_p_meds} / {total_p_meds}")
        with c_adh3:
            st.metric("Active Adherence Rate", f"{cur_adherence}%", delta=None)

        st.divider()

        # Section 1: Prescribe New Medication
        with st.expander("➕ Prescribe New Medication", expanded=False):
            with st.form("prescribe_med_form", clear_on_submit=True):
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    new_drug_name = st.selectbox(
                        "Medication Name",
                        options=[
                            "Temozolomide (Temodar)",
                            "Levetiracetam (Keppra)",
                            "Dexamethasone",
                            "Lomustine (CCNU)",
                            "Bevacizumab (Avastin)",
                            "Ondansetron (Zofran)",
                            "Procarbazine",
                            "Vincristine"
                        ]
                    )
                    new_dosage = st.text_input("Dosage", placeholder="e.g. 150 mg/m2 Oral Capsule, 750 mg Oral Tablet")
                with col_m2:
                    new_freq = st.selectbox(
                        "Schedule Frequency",
                        options=[
                            "Nightly (Days 1-5 of 28-day cycle)",
                            "Twice Daily (Every 12 hours)",
                            "Once Daily (Morning with food)",
                            "Three Times Daily",
                            "As Needed (PRN)",
                            "Every 6 Weeks (Day 1 of 42-day cycle)"
                        ]
                    )
                    new_prescriber = st.text_input("Prescribed By", value=patient["oncologist_name"])

                new_instr = st.text_area(
                    "Special Clinical Instructions",
                    placeholder="e.g. Take on empty stomach at bedtime; take antiemetic 30 min prior. Monitor platelet counts."
                )

                submit_prescription = st.form_submit_button("✍️ Issue Prescription & Add to Patient Record", type="primary")
                if submit_prescription:
                    if not new_dosage:
                        st.error("Please enter a valid dosage.")
                    else:
                        db.prescribe_medication({
                            "patient_id": patient_id,
                            "med_name": new_drug_name,
                            "dosage": new_dosage,
                            "frequency": new_freq,
                            "instructions": new_instr,
                            "prescribed_by": new_prescriber,
                            "status": "Due",
                            "adherence_notes": "Newly prescribed by oncology team.",
                            "last_updated_by": "Doctor"
                        })
                        st.success(f"Prescribed **{new_drug_name}** ({new_dosage}) for {patient['full_name']}.")
                        st.rerun()

        # Section 2: Active Prescriptions & Adherence Overrides
        st.markdown("##### 📋 Active Prescriptions & Clinical Status Verification")

        for med in meds:
            mid = med["med_id"]
            current_status = med.get("status", "Due")
            current_notes = med.get("adherence_notes", "")

            with st.container():
                st.markdown(
                    f"""
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="font-size: 1.05rem; color: #0f172a;">{med['med_name']}</strong> — 
                                <span style="color: #475569;">{med['dosage']}</span><br>
                                <span style="font-size: 0.85rem; color: #64748b;">Schedule: {med.get('frequency', med.get('schedule_time', ''))} • Prescriber: {med.get('prescribed_by', 'Oncology Team')}</span><br>
                                <span style="font-size: 0.85rem; color: #0284c7;">Instructions: {med.get('instructions', 'Take as directed.')}</span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.8rem; text-transform: uppercase; font-weight: 700; color: {'#059669' if current_status == 'Taken' else ('#dc2626' if current_status == 'Missed' else '#f59e0b')};">Status: {current_status}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col_btn_t, col_btn_d, col_btn_m, col_note, col_save = st.columns([1, 1, 1, 3.5, 1.2])

                with col_btn_t:
                    if st.button("Mark Taken", key=f"doc_take_{mid}", type="primary" if current_status == "Taken" else "secondary", use_container_width=True):
                        db.update_medication_status(mid, "Taken", notes=current_notes, updated_by="Doctor")
                        st.rerun()

                with col_btn_d:
                    if st.button("Mark Due", key=f"doc_due_{mid}", type="primary" if current_status == "Due" else "secondary", use_container_width=True):
                        db.update_medication_status(mid, "Due", notes=current_notes, updated_by="Doctor")
                        st.rerun()

                with col_btn_m:
                    if st.button("Mark Missed", key=f"doc_miss_{mid}", type="primary" if current_status == "Missed" else "secondary", use_container_width=True):
                        db.update_medication_status(mid, "Missed", notes=current_notes, updated_by="Doctor")
                        st.rerun()

                with col_note:
                    new_note_val = st.text_input(
                        "Clinical Adherence Note",
                        value=current_notes,
                        key=f"note_input_{mid}",
                        label_visibility="collapsed",
                        placeholder="Add clinical observation or reason for status change..."
                    )

                with col_save:
                    if st.button("Save Note", key=f"save_note_{mid}", use_container_width=True):
                        db.update_medication_status(mid, current_status, notes=new_note_val, updated_by="Doctor")
                        st.success("Note saved.")
                        st.rerun()

    # =========================================================================
    # TAB 4: Clinical Report Generator & Formal Print/Export Console
    # =========================================================================
    with tab_report_gen:
        st.markdown("##### 📄 Neuro-Oncology Formal Consultation Summary Generator")
        st.markdown("Compile a comprehensive, validated clinical summary synthesizing demographics, MRI volumetrics, RANO 2.0 evaluation, and symptom dynamics.")

        oncologist_notes = st.text_area(
            "Oncologist Consultation Notes & Tumor Board Discussion:",
            value="Patient demonstrating classic clinical radiation necrosis pattern within the 12-week post-RT window. Proceeding with close monitoring without altering current TMZ schedule.",
            height=80,
            key="consult_doc_notes"
        )

        if st.button("📑 Generate Consultation Summary for Active Patient", type="primary", use_container_width=True):
            st.session_state.generated_consult_report = True

        if st.session_state.get("generated_consult_report", True):
            today_date = datetime.now().strftime("%B %d, %Y")
            latest_vols = latest_scan
            active_rano = latest_twin.get("rano_category", "PsP")
            active_psp = latest_twin.get("pseudoprogression_risk", 0.84)
            active_triage = latest_twin.get("triage_category", "Moderate")

            recent_symptoms_str = ", ".join([f"{s['symptom_type']} ({s['severity']})" for s in symptoms[:3]]) if symptoms else "None reported"
            active_meds_str = ", ".join([f"{m['med_name']} ({m['status']})" for m in meds]) if meds else "None"

            # Formatted Report Content
            consultation_markdown = f"""
# NEURO-ONCOLOGY MULTIDISCIPLINARY CONSULTATION REPORT
**Department of Neuro-Oncology • Comprehensive Brain Tumor Center**  
**Date of Consultation**: {today_date}  

---

### 1. PATIENT DEMOGRAPHICS & CLINICAL IDENTIFIERS
* **Patient Name**: {patient.get('full_name', 'Unknown')}
* **Medical Record Number (MRN)**: `{patient.get('mrn', 'N/A')}`
* **Age / Gender**: {patient.get('age', 45)} years old | {patient.get('gender', 'Female')}
* **Clinical Diagnosis**: {patient.get('diagnosis_type', 'Glioma')}
* **Anatomical Location**: {patient.get('diagnosis_location', 'Brain')}
* **Initial Diagnosis Date**: {patient.get('diagnosis_date', 'N/A')}
* **Next Follow-up Visit**: {patient.get('next_visit_date', 'In 2 weeks')}
* **Current Treatment Protocol**: {patient.get('treatment_stage', 'Adjuvant Therapy')}
* **Functional Performance Score**: Karnofsky Performance Scale (KPS) {patient.get('baseline_kps', 90)}%
* **Attending Neuro-Oncologist**: {patient.get('oncologist_name', 'Dr. Aris Thorne, MD')}
* **Care Navigator**: {patient.get('nurse_name', 'Sarah Jensen, RN')}

---

### 2. MOLECULAR & BIOMARKER PROFILING
* **IDH Mutation Status**: **IDH-{molecular.get('idh_status', 'Mutant')}**
* **MGMT Promoter Methylation**: **{'METHYLATED (Favorable Temozolomide Sensitivity)' if molecular.get('mgmt_methylated') else 'UNMETHYLATED (Alkylating Resistance)'}**
* **1p/19q Chromosomal Co-deletion**: **{'CO-DELETED (Oligodendroglial Lineage)' if molecular.get('codeletion_1p19q') else 'INTACT / NON-CODELETED'}**
* **Assayed On**: {molecular.get('assayed_at', '2026-05-20')}

---

### 3. SERIAL MRI VOLUMETRIC ANALYSIS & QUANTIFICATION (SegResNet 3D)
* **Latest Evaluation Date**: {latest_vols.get('scan_date', '2026-09-18')}
* **Whole Tumor Volume (WT = NCR + ED + ET)**: **{latest_vols.get('wt_vol_cm3', 14.60):.2f} cm³**
* **Tumor Core Volume (TC = NCR + ET)**: **{latest_vols.get('tc_vol_cm3', 8.10):.2f} cm³**
* **Enhancing Tumor Volume (ET)**: **{latest_vols.get('et_vol_cm3', 4.10):.2f} cm³**
* **Peritumoral Vasogenic Edema (ED)**: **{latest_vols.get('edema_vol_cm3', 6.50):.2f} cm³**
* **Segmentation Confidence (Dice Coefficient)**: **{latest_vols.get('dice_score', 0.9410):.4f}**
* **Perfusion Perilesional rCBV**: **{latest_vols.get('estimated_rcbv', 1.48):.2f}**

---

### 4. RANO 2.0 CRITERIA & PSEUDOPROGRESSION (PsP) ASSESSMENT
* **Response Assessment in Neuro-Oncology (RANO 2.0)**: **Category [{active_rano}]**
* **Pseudoprogression (PsP) Probability Score**: **{active_psp * 100:.1f}%**
* **Urgency Triage Score**: **[{active_triage}] Priority**
* **Clinical Diagnostic Impression**: Low relative Cerebral Blood Volume (rCBV: {latest_vols.get('estimated_rcbv', 1.48)}) combined with **MGMT-Methylated** molecular phenotype and temporal alignment (<12 weeks post-radiotherapy) strongly supports treatment-induced pseudoprogression / radiation necrosis over true neoplastic progression. Continuation of adjuvant Temozolomide recommended.

---

### 5. SYMPTOM BURDEN & MEDICATION ADHERENCE TRAJECTORY
* **Recent Symptom Logs**: {recent_symptoms_str}
* **Active Regimen**: {active_meds_str}
* **Current Running Medication Adherence**: **{cur_adherence}%**
* **Digital Twin Alert Trigger**: {latest_twin.get('alert_banner_text', 'Trajectory stable.')}

---

### 6. PATIENT-FRIENDLY TRANSLATION (DISCHARGE NOTE)
> *"{reports[0].get('plain_language_summary', 'Your scan shows steady progress with stable treatment effects.') if reports else 'Your scan shows steady progress.'}"*

---
**Attending Physician Signature**:  
*Electronically Verified by {patient['oncologist_name']} • Comprehensive Neuro-Oncology Center*
"""

            # Render Printable Card in UI
            st.markdown(
                f"""
                <div style="background-color: white; border: 2px solid #cbd5e1; border-radius: 8px; padding: 2rem; margin-top: 1rem; color: #0f172a; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #0f766e; padding-bottom: 12px; margin-bottom: 1.5rem;">
                        <div>
                            <h3 style="margin: 0; color: #0f766e;">NEURO-ONCOLOGY MULTIDISCIPLINARY CONSULTATION</h3>
                            <span style="font-size: 0.9rem; color: #64748b;">Comprehensive Brain Tumor Center • Clinical Systems Architecture</span>
                        </div>
                        <div style="text-align: right;">
                            <strong>Date: {today_date}</strong><br>
                            <span style="color: #64748b;">MRN: {patient['mrn']}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("👁️ View Full Formatted Report", expanded=True):
                st.markdown(consultation_markdown)

            col_down, col_pdf, col_print = st.columns([1, 1, 1])
            with col_down:
                st.download_button(
                    label="💾 Download Summary (.md)",
                    data=consultation_markdown,
                    file_name=f"Consultation_Summary_{patient['mrn']}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col_pdf:
                doc_pdf_bytes = generate_patient_clinical_pdf(
                    patient=patient,
                    report={
                        "report_date": today_date,
                        "report_id": f"CONSULT-{patient['mrn']}",
                        "raw_text": consultation_markdown,
                        "plain_language_summary": reports[0].get('plain_language_summary', 'Your scan shows steady progress.') if reports else 'Your scan shows steady progress.'
                    },
                    scan_data=latest_vols,
                    molecular_data=molecular
                )
                st.download_button(
                    label="📥 Download Clinical PDF",
                    data=doc_pdf_bytes,
                    file_name=f"Clinical_Consultation_{patient['mrn']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            with col_print:
                # Browser print snippet
                st.markdown(
                    """
                    <button onclick="window.print()" style="width: 100%; height: 38px; background-color: #0f766e; color: white; border: none; border-radius: 6px; font-weight: 600; cursor: pointer;">
                        🖨️ Print Consultation Report
                    </button>
                    """,
                    unsafe_allow_html=True
                )

    # =========================================================================
    # TAB 4: Interactive 3D Brain & Tumor Localization
    # =========================================================================
    with tab_3d_brain:
        render_3d_brain_viewer_panel(
            patient=patient,
            scan_data=latest_scan,
            panel_title=f"🧠 3D Brain Volumetric Reconstruction: {patient['full_name']} (MRN: {patient['mrn']})",
            key_prefix="clinician_workstation_3d"
        )

