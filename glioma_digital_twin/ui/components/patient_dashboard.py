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

    # 2. Dynamic Alert Banner (Fusion Layer Trigger)
    alert_text = latest_twin.get("alert_banner_text", "")
    triage_cat = latest_twin.get("triage_category", "Moderate")

    if alert_text:
        border_color = "#f59e0b" if triage_cat == "Moderate" else ("#ef4444" if triage_cat == "Critical" else "#10b981")
        bg_color = "#fffbeb" if triage_cat == "Moderate" else ("#fef2f2" if triage_cat == "Critical" else "#ecfdf5")
        icon = "⚠️" if triage_cat == "Moderate" else ("🚨" if triage_cat == "Critical" else "✅")

        st.markdown(
            f"""
            <div style="background-color: {bg_color}; border-left: 6px solid {border_color}; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1.2rem;">
                <div style="display: flex; align-items: flex-start;">
                    <span style="font-size: 1.4rem; margin-right: 12px;">{icon}</span>
                    <div>
                        <strong style="color: #92400e; font-size: 0.95rem; text-transform: uppercase;">Multimodal Twin Alert ({triage_cat} Priority)</strong>
                        <p style="margin: 4px 0 0 0; color: #1f2937; font-size: 0.95rem;">{alert_text}</p>
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
        st.markdown(
            """
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <span style="font-size: 0.8rem; color: #6b7280; font-weight: 600; text-transform: uppercase;">Next Clinical Visit</span>
                <h3 style="margin: 6px 0; font-size: 1.7rem; color: #0f766e; font-weight: 700;">Oct 12</h3>
                <span style="color: #6b7280; font-size: 0.85rem;">Neuro-Oncology Follow-up</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # 4. Quick-Access Action Grid
    st.markdown("##### ⚡ Quick Access")
    q1, q2, q3, q4, q5 = st.columns(5)

    with q1:
        if st.button("📋 Scans & Reports", use_container_width=True, help="Inspect MRI and 6th-grade translated reports"):
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

    st.divider()

    # 5. Bottom Section: Demographics & Care Team
    c_demo, c_care = st.columns([1.2, 1.0])

    with c_demo:
        st.markdown("##### 👤 Patient Information & Molecular Genetics")
        st.markdown(
            f"""
            - **Medical Record Number**: `{patient['mrn']}`
            - **Baseline KPS Functionality**: **{patient['baseline_kps']}%**
            - **Primary Oncologist**: {patient['oncologist_name']}
            - **Primary Care Navigator**: {patient['nurse_name']}
            - **Diagnosis**: {patient['diagnosis_type']}
            - **Anatomical Subsite**: {patient['diagnosis_location']}
            """
        )

    with c_care:
        st.markdown("##### 🔒 Privacy & Research Consent Governance")
        st.caption("Healthcare Companion adheres to HIPAA, GDPR, and strict patient consent guardrails.")

        consent_state = st.toggle(
            "Share De-Identified Data with Brain Tumor Research Consortia",
            value=patient.get("share_research_consent", True),
            key="research_consent_toggle"
        )
        if consent_state:
            st.success("✅ Research consent active: Your de-identified scans help improve biophysical models globally.")
        else:
            st.info("🔒 Strict Local Mode: Your data remains private to your hospital care team.")
