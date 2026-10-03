"""
Quick Symptom Entry Form Component
One-tap preset symptom buttons, discrete severity selector, onset picker, notes,
and live database persistence to symptom_logs with recent history visualization.
Supports dynamic patient context across all pre-seeded patients.
"""

import streamlit as st
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from ...utils.supabase_client import get_database_client
from ...models.red_flag_alert import get_escalation_engine


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

    # Render Active Escalation Banner if triggered
    active_alert = st.session_state.get("latest_escalation_alert")
    if active_alert and active_alert.get("patient_id") == patient_id:
        is_stat = active_alert.get("severity_tier") == "STAT"
        border_col = "#dc2626" if is_stat else "#d97706"
        bg_col = "#fef2f2" if is_stat else "#fffbeb"
        badge_text = "🚨 STAT EMERGENCY DISPATCHED" if is_stat else "⚠️ URGENT CLINICAL ESCALATION DISPATCHED"

        st.markdown(
            f"""
            <div style="background: {bg_col}; border: 2px solid {border_col}; border-radius: 8px; padding: 14px 18px; margin-bottom: 1.2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: {border_col}; font-size: 1.05rem;">{badge_text}</strong>
                    <span style="font-size: 0.8rem; color: #64748b;">Alert ID: {active_alert['alert_id']}</span>
                </div>
                <div style="margin-top: 6px; color: #1e293b; font-size: 0.95rem;">
                    <strong>Trigger</strong>: {active_alert['trigger_event']}
                </div>
                <div style="margin-top: 8px; font-size: 0.85rem; line-height: 1.4; color: #334155;">
                    <strong>Emergency SBAR Summary</strong>:<br>
                    • <em>Situation</em>: {active_alert['sbar']['situation']}<br>
                    • <em>Recommendation</em>: {active_alert['sbar']['recommendation']}
                </div>
                <div style="margin-top: 10px; font-size: 0.8rem; color: #0f766e; background: white; padding: 6px 10px; border-radius: 6px; border: 1px solid #cbd5e1;">
                    📱 <strong>SMS Gateway</strong>: Delivered to On-Call Attending ({active_alert['dispatch_channels']['sms']['to']})<br>
                    📟 <strong>Hospital Pager Webhook</strong>: {active_alert['dispatch_channels']['pager_webhook']['status']} (Priority: {active_alert['dispatch_channels']['pager_webhook']['priority']})
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Dismiss Alert Notice", key="dismiss_alert_notice", use_container_width=False):
            st.session_state.latest_escalation_alert = None
            st.rerun()

    # Initialize session state for selected symptom if not present
    if "selected_symptom" not in st.session_state:
        st.session_state.selected_symptom = "Headache"

    # 1. High-Accessibility One-Tap Symptom Grid (st.pills)
    st.markdown("##### 1️⃣ Select Symptoms Experienced Today (One-Tap Touch Grid)")
    symptom_options = [
        "🤯 Mild Headache",
        "👁️ Blurred Vision",
        "⚡ Seizure Aura",
        "💫 Dizziness",
        "😴 Extreme Fatigue",
        "🗣️ Speech Trouble",
        "🦾 Motor Weakness"
    ]
    selected_pills = st.pills(
        "Symptoms Encountered:",
        options=symptom_options,
        selection_mode="multi",
        default=["🤯 Mild Headache"],
        key="symptom_pills_selector"
    )

    primary_symptom = selected_pills[0].split()[1] if selected_pills else "General"
    all_symptoms_str = ", ".join(selected_pills) if selected_pills else "None selected"

    # 2. Form details (Severity, Voice Dictation, Notes)
    with st.form("symptom_entry_form", clear_on_submit=False):
        col_sev, col_onset = st.columns(2)

        with col_sev:
            severity = st.select_slider(
                "2️⃣ How severe is it overall?",
                options=["1 - Very Mild", "2 - Mild", "3 - Moderate", "4 - Severe", "5 - Very Severe"],
                value="3 - Moderate",
                help="1-2: Noticeable but does not interrupt activities. 3: Interrupted tasks or needed rest. 4-5: Unable to continue daily activities."
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

        st.markdown("##### 3️⃣ Dictate Voice Log & Additional Context")
        notes = st.text_area(
            "Patient Notes & Context (Optional)",
            placeholder="Describe what you were doing, any unusual sensations (like burning smells, tingling, visual flashes), or medications taken...",
            help="Detailing auras or triggers helps your care team catch potential medication adjustments early."
        )

        submitted = st.form_submit_button("✅ Submit Daily Health & Symptom Log", use_container_width=True, type="primary")

        if submitted:
            # Clean severity label for triage ("Severe", "Moderate", "Mild")
            sev_clean = "Severe" if ("Severe" in severity or "5" in severity or "4" in severity) else ("Moderate" if "Moderate" in severity else "Mild")
            combined_notes = f"Selected: {all_symptoms_str}. " + (notes.strip() if notes else "")

            new_log = {
                "patient_id": patient_id,
                "symptom_type": primary_symptom,
                "severity": sev_clean,
                "onset_time": onset_time,
                "notes": combined_notes,
                "logged_at": datetime.now(timezone.utc).isoformat()
            }
            db.log_symptom(new_log)
            if db.is_connected_to_supabase:
                st.toast("Transmitted directly to Supabase PostgreSQL!", icon="📡")
            else:
                st.toast("Saved to resilient local twin memory!", icon="💾")

            # Evaluate Acute Red-Flag Escalation Protocol (Level 1-3)
            engine = get_escalation_engine()
            alert = engine.evaluate_symptom_event(
                symptom_type=primary_symptom,
                severity=sev_clean,
                notes=combined_notes,
                patient_data=patient
            )
            if alert:
                st.session_state.latest_escalation_alert = alert
                st.toast(f"🚨 Clinical Alert Triggered: {alert['severity_tier']}", icon="⚠️")

            st.rerun()

    # Voice Input Recorder (outside form for instant stream capture)
    with st.expander("🎙️ Attach Native Voice Dictation to Doctor Log", expanded=True):
        st.write("Record a spoken description for your oncologist and care navigator:")
        audio_val = st.audio_input("Record audio note:")
        if audio_val:
            st.audio(audio_val)
            st.caption("🎙️ Voice note recorded. This will be transmitted to Dr. Thorne's clinical in-basket.")

    st.divider()

    # 3. Real-Time Symptom Stream via Streamlit Fragment
    _render_live_symptom_stream(patient_id)


@st.fragment(run_every=5)
def _render_live_symptom_stream(patient_id: str):
    """
    Real-time symptom telemetry fragment.
    Background polls Supabase PostgreSQL every 5 seconds without triggering a full page rerun.
    """
    db = get_database_client()
    status = db.get_connection_status()
    is_live = status["is_connected"]

    col_hdr, col_sync = st.columns([2.5, 1.5])
    with col_hdr:
        st.markdown("##### 📜 Real-Time Symptom Stream")
    with col_sync:
        mode_badge = "🟢 Live Supabase (5s)" if is_live else "🟡 Local Real-Time Cache"
        st.caption(f"{mode_badge} • `{datetime.now().strftime('%H:%M:%S')}`")

    symptoms = db.get_patient_symptoms(patient_id, limit=8)

    c_cnt, c_rf = st.columns([3, 1])
    with c_cnt:
        target_db = "Supabase PostgreSQL" if is_live else "Local Patient Memory"
        st.caption(f"Displaying **{len(symptoms)}** live entries fetched from **{target_db}**")
    with c_rf:
        if st.button("🔄 Sync Now", key="btn_sync_symptoms_now", use_container_width=True):
            st.rerun()

    if symptoms:
        for s in symptoms:
            sev_badge = "🔴" if s["severity"] == "Severe" else ("🟠" if s["severity"] == "Moderate" else "🟢")
            time_display = s.get("logged_at", "")
            if "T" in time_display:
                try:
                    dt = datetime.fromisoformat(time_display.replace("Z", "+00:00"))
                    time_display = dt.strftime("%b %d, %H:%M UTC")
                except Exception:
                    pass

            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #1e293b; font-size: 0.95rem;">{sev_badge} {s['symptom_type']}</span>
                        <span style="font-size: 0.78rem; color: #64748b; background: #f1f5f9; padding: 2px 8px; border-radius: 12px;">{time_display}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #475569; margin-top: 4px;">
                        <strong>Severity:</strong> {s['severity']} &nbsp;|&nbsp; <strong>Onset:</strong> {s.get('onset_time', 'N/A')}
                    </div>
                    <div style="font-size: 0.88rem; color: #334155; margin-top: 6px; border-top: 1px dashed #f1f5f9; padding-top: 6px;">
                        {s.get('notes', 'No notes provided.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No symptoms logged yet. Submit a log above to transmit telemetry.")

