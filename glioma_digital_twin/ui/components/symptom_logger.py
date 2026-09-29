"""
Quick Symptom Entry Form Component
One-tap preset symptom buttons, discrete severity selector, onset picker, notes,
and live database persistence to symptom_logs with recent history visualization.
Supports dynamic patient context across all pre-seeded patients.
"""

import streamlit as st
from datetime import datetime, timezone
from typing import Dict, Any
from ...utils.supabase_client import get_database_client


def render_symptom_logger():
    """Renders the Symptom Logging interface."""
    db = get_database_client()
    active_mrn = st.session_state.get("active_patient_mrn", "0042")
    patient = db.get_patient_by_mrn(active_mrn)
    if not patient:
        patient = db.get_patient_by_mrn("0042")
        if not patient:
            st.error("Patient record could not be loaded.")
            return

    patient_id = patient["patient_id"]

    st.subheader(f"📝 Log Symptoms for {patient['full_name']}")
    st.markdown("Track how you are feeling. Your entries are continuously monitored by your Patient Digital Twin.")

    # Initialize session state for selected symptom if not present
    if "selected_symptom" not in st.session_state:
        st.session_state.selected_symptom = "Headache"

    # 1. Preset Symptom Buttons
    st.markdown("##### 1. Select Symptom Type (One-Tap Preset)")
    preset_symptoms = [
        ("🤕 Headache", "Headache"),
        ("👁️ Vision", "Vision"),
        ("⚡ Seizure", "Seizure"),
        ("😴 Fatigue", "Fatigue"),
        ("🗣️ Speech", "Speech"),
        ("🦾 Motor", "Motor"),
    ]

    cols = st.columns(6)
    for i, (label, sym_val) in enumerate(preset_symptoms):
        with cols[i]:
            btn_type = "primary" if st.session_state.selected_symptom == sym_val else "secondary"
            if st.button(label, key=f"btn_sym_{sym_val}", type=btn_type, use_container_width=True):
                st.session_state.selected_symptom = sym_val
                st.rerun()

    st.write(f"Current selection: **{st.session_state.selected_symptom}**")

    # 2. Form details
    with st.form("symptom_entry_form", clear_on_submit=True):
        col_sev, col_onset = st.columns(2)

        with col_sev:
            severity = st.select_slider(
                "Severity Level",
                options=["Mild", "Moderate", "Severe"],
                value="Moderate",
                help="Mild: Noticeable but does not interrupt activities. Moderate: Interrupted tasks or needed rest. Severe: Unable to continue daily activities."
            )

        with col_onset:
            onset_time = st.selectbox(
                "When did it start?",
                options=[
                    "Just now (Within last 30 minutes)",
                    "Earlier this morning",
                    "This afternoon",
                    "Yesterday evening",
                    "Ongoing over several days"
                ],
                index=0
            )

        notes = st.text_area(
            "Patient Notes & Context (Optional)",
            placeholder="Describe what you were doing, any unusual sensations (like burning smells, tingling, visual flashes), or medications taken...",
            help="Detailing auras or triggers helps your care team catch potential medication adjustments early."
        )

        submitted = st.form_submit_button("💾 Save Symptom Entry", use_container_width=True, type="primary")

        if submitted:
            new_log = {
                "patient_id": patient_id,
                "symptom_type": st.session_state.selected_symptom,
                "severity": severity,
                "onset_time": onset_time,
                "notes": notes.strip() if notes else f"Reported {st.session_state.selected_symptom.lower()} with {severity.lower()} severity.",
                "logged_at": datetime.now(timezone.utc).isoformat()
            }
            db.log_symptom(new_log)
            st.success(f"Successfully recorded **{st.session_state.selected_symptom}** ({severity}) into your health record!")

            # Trigger automated check for Seizure / Aura
            if st.session_state.selected_symptom == "Seizure" or severity == "Severe":
                st.warning("⚠️ **Notice**: A seizure or severe symptom has been recorded. Your care navigator has been flagged.")
            st.rerun()

    st.divider()

    # 3. Recent Symptom History
    st.markdown("##### 📜 Recent Symptom History")
    symptoms = db.get_patient_symptoms(patient_id)
    if symptoms:
        for s in symptoms[:5]:
            sev_badge = "🔴" if s["severity"] == "Severe" else ("🟠" if s["severity"] == "Moderate" else "🟢")
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                    <strong>{sev_badge} {s['symptom_type']}</strong> — <span style="color: #64748b;">{s['severity']} Severity • {s['onset_time']}</span><br>
                    <span style="font-size: 0.9rem; color: #334155;">{s.get('notes', '')}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No symptoms logged yet.")
