"""
Patient Dashboard Component
Dynamic Alert Banner, 4 Core Stat Cards, Quick-Access Action Grid,
Demographics, Care Team Profile, and Privacy / Research Consent Toggles.
Supports dynamic patient context across all pre-seeded patients.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Callable
from ...utils.supabase_client import get_database_client


def render_patient_dashboard(on_navigate: Callable[[str], None]):
    """Renders Patient Portal Home Dashboard."""
    db = get_database_client()
    active_mrn = st.session_state.get("active_patient_mrn", "0042")
    patient = db.get_patient_by_mrn(active_mrn)
    if not patient:
        patient = db.get_patient_by_mrn("0042")
        if not patient:
            st.error("Patient record could not be loaded.")
            return

    patient_id = patient["patient_id"]
    scans = db.get_patient_scans(patient_id)
    reports = db.get_patient_reports(patient_id)
    meds = db.get_patient_medications(patient_id)
    symptoms = db.get_patient_symptoms(patient_id)
    timeline = db.get_patient_timeline(patient_id)

    latest_scan = scans[-1] if scans else {}
    prior_scan = scans[-2] if len(scans) >= 2 else scans[-1] if scans else {}
    latest_twin = timeline[0] if timeline else {}

    # Calculate adherence
    total_meds = len(meds)
    taken_meds = sum(1 for m in meds if m.get("status") == "Taken")
    adherence_pct = round((taken_meds / max(1, total_meds)) * 100.0, 1)

    # 1. Header with Patient Welcome
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%); padding: 1.4rem; border-radius: 12px; color: white; margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700;">Welcome back, {patient['full_name']}</h2>
                    <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 0.95rem;">
                        MRN: <strong>{patient['mrn']}</strong> • Age: {patient['age']} • {patient['diagnosis_type']} ({patient['diagnosis_location']})
                    </p>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 8px;">
                    <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Current Stage</span><br>
                    <strong>{patient['treatment_stage']}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



    # 3. Four Core Stat Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        wt_vol = float(latest_scan.get("wt_vol_cm3", 14.60))
        prior_wt = float(prior_scan.get("wt_vol_cm3", wt_vol))
        delta_v = wt_vol - prior_wt
        delta_str = f"▼ {abs(delta_v):.1f} cm³" if delta_v < 0 else (f"▲ +{delta_v:.1f} cm³" if delta_v > 0 else "Stable")
        delta_col = "#059669" if delta_v <= 0 else "#dc2626"
        scan_sub = f"{delta_str} • {len(scans)} scans"
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <span style="font-size: 0.8rem; color: #6b7280; font-weight: 600; text-transform: uppercase;">Tumor Volume</span>
                <h3 style="margin: 6px 0; font-size: 1.7rem; color: #0f766e; font-weight: 700;">{wt_vol:.1f} <span style="font-size: 1rem; font-weight: normal; color: #6b7280;">cm³</span></h3>
                <span style="color: {delta_col}; font-size: 0.85rem; font-weight: 600;">{scan_sub}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        adh_color = "#059669" if adherence_pct >= 80 else "#dc2626"
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <span style="font-size: 0.8rem; color: #6b7280; font-weight: 600; text-transform: uppercase;">Medication Adherence</span>
                <h3 style="margin: 6px 0; font-size: 1.7rem; color: {adh_color}; font-weight: 700;">{adherence_pct}%</h3>
                <span style="color: #6b7280; font-size: 0.85rem;">{taken_meds} of {total_meds} verified doses</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        sync_tag = "🟢 Live Supabase Sync" if db.is_connected_to_supabase else "🟡 Local Twin Memory"
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <span style="font-size: 0.8rem; color: #6b7280; font-weight: 600; text-transform: uppercase;">Symptoms Logged</span>
                <h3 style="margin: 6px 0; font-size: 1.7rem; color: #0f766e; font-weight: 700;">{len(symptoms)}</h3>
                <span style="color: #059669; font-size: 0.82rem;">{sync_tag}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:
        next_visit_str = patient.get("next_visit_date", "28-Oct-2026")
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <span style="font-size: 0.8rem; color: #6b7280; font-weight: 600; text-transform: uppercase;">Next Clinical Visit</span>
                <h3 style="margin: 6px 0; font-size: 1.5rem; color: #0f766e; font-weight: 700;">{next_visit_str}</h3>
                <span style="color: #6b7280; font-size: 0.85rem;">Neuro-Oncology Follow-up</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # 4. Quick-Access Action Grid
    st.markdown("##### ⚡ Quick Access")
    q1, q2, q3, q4, q5, q6 = st.columns(6)

    with q1:
        if st.button("📋 Scans & Reports", use_container_width=True, help="Inspect MRI and 10-year-old kid translated reports"):
            on_navigate("reports")
    with q2:
        if st.button("📝 Log Symptom", use_container_width=True, help="Record headaches, seizures, or fatigue"):
            on_navigate("symptoms")
    with q3:
        if st.button("💊 Medications", use_container_width=True, help="Track and verify medication adherence"):
            on_navigate("companion")
    with q4:
        if st.button("💬 Care Companion", use_container_width=True, help="Chat with grounded conversational assistant"):
            on_navigate("companion")
    with q5:
        if st.button("📤 Upload MRI Scan", use_container_width=True, help="Upload and evaluate brain scan"):
            on_navigate("evaluator")
    with q6:
        if st.button("📄 Upload & Verify Reports", use_container_width=True, help="Upload and verify clinical text reports"):
            on_navigate("report_verify")

    st.divider()

    # 5. Bottom Section: Demographics & Care Team
    c_demo, c_care = st.columns([1.2, 1.0])

    with c_demo:
        st.markdown("##### 👤 Patient Information & Molecular Genetics")
        st.markdown(
            f"""
            - **Medical Record Number**: `{patient.get('mrn', 'N/A')}`
            - **Baseline KPS Functionality**: **{patient.get('baseline_kps', 90)}%**
            - **Primary Oncologist**: {patient.get('oncologist_name', 'Dr. Aris Thorne, MD')}
            - **Primary Care Navigator**: {patient.get('nurse_name', 'Sarah Jensen, RN')}
            - **Diagnosis**: {patient.get('diagnosis_type', 'Glioma')}
            - **Anatomical Subsite**: {patient.get('diagnosis_location', 'Brain')}
            - **Initial Diagnosis Date**: `{patient.get('diagnosis_date', 'N/A')}`
            """
        )

    with c_care:
        st.markdown("##### 🏥 Care Team & Clinical Contacts")
        st.markdown(
            f"""
            - **Attending Neuro-Oncologist**: {patient.get('oncologist_name', 'Dr. Aris Thorne, MD')}
            - **Nurse Care Navigator**: {patient.get('nurse_name', 'Sarah Jensen, RN')}
            - **Next Follow-up Visit**: `{next_visit_str}`
            - **Neuro-Oncology Triage Helpline**: `+91 (080) 4123-8900`
            - **On-Call Attending Line**: `+1 (555) 019-8492`
            - **Location**: Comprehensive Brain Tumor Center & Clinic
            """
        )
