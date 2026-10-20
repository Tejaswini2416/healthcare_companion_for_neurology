"""
Personalized Conversational AI Healthcare Companion Component
Features:
1. One-tap prescription adherence tracking and live compliance analytics.
2. Full ChatGPT-like interactive Healthcare Companion grounded in the Patient Digital Twin.
3. Natural language conversation for symptoms, diet, wellness, medications, and clinical questions.
4. Emergency triage detection with prominent India (112 / 108) and international emergency numbers.
5. Medical safety safeguards: strict prescription refusal and evidence-based non-diagnostic education.
6. Clinical FAQ & structured guidance (Healthy Diet, When to See a Doctor, Appointment Preparation).
7. Real-time appointment assistance and schedule viewing from the patient's record.
8. Model engine switcher (OpenAI GPT-4o, Groq LLaMA-3.3, Google Gemini, and Built-in Clinical Bio-LLM).
9. Multi-turn conversation memory, history reset, and transcript export controls.
10. GraphRAG anti-hallucination verification badges and clinical citations.
"""

import os
import json
import streamlit as st
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from ...utils.supabase_client import get_database_client
from ...models.companion_agent import CareCompanionAgent


def _clean_content(text: str) -> str:
    """Strips repetitive in-bubble disclaimer text and redundant greetings for a clean conversational UI."""
    if not text:
        return ""
    if "⚠️ *HealthMate Medical Disclaimer" in text:
        text = text.split("⚠️ *HealthMate Medical Disclaimer")[0].strip()
    if "---\n⚠️" in text:
        text = text.split("---\n⚠️")[0].strip()
    return text


def _render_graph_rag_badge(rag_data: Optional[Dict[str, Any]]):
    """UI display disabled per user request for clean clinical interface."""
    return


def _render_emergency_callout_banner():
    """Renders persistent emergency guidance and emergency hotline banner."""
    st.markdown(
        """
        <div style="background: #fef2f2; border: 1.5px solid #f87171; border-radius: 10px; padding: 12px 16px; margin-bottom: 1rem;">
            <div style="display: flex; align-items: flex-start; gap: 12px;">
                <span style="font-size: 1.6rem;">🚨</span>
                <div style="flex: 1;">
                    <div style="font-weight: 700; color: #991b1b; font-size: 0.95rem; margin-bottom: 2px;">
                        Immediate Medical Emergency Guidance
                    </div>
                    <div style="font-size: 0.82rem; color: #7f1d1d; line-height: 1.45;">
                        If you are experiencing sudden weakness, chest pain, paralysis, difficulty breathing, or the worst headache of your life, please seek immediate emergency care:
                    </div>
                    <div style="margin-top: 6px; display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.8rem;">
                        <span style="background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 6px; font-weight: 700;">
                            🇮🇳 India: Dial 112 (All Emergencies) / 108 (Ambulance)
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_companion_chat(show_meds: bool = True):
    """
    Renders the full Conversational Care Companion GPT, Medication Tracker,
    and Appointment Assistance workspace.
    """
    db = get_database_client()
    active_mrn = st.session_state.get("active_patient_mrn", "0042")
    patient = db.get_patient_by_mrn(active_mrn)
    if not patient:
        patient = db.get_patient_by_mrn("0042")
        if not patient:
            st.error("Patient record could not be loaded.")
            return

    patient_id = patient["patient_id"]
    meds = db.get_patient_medications(patient_id)
    appointments = db.get_patient_appointments(patient_id)
    first_name = patient["full_name"].split()[0] if patient.get("full_name") else "there"

    # =========================================================================
    # 1. OPTIONAL: MEDICATIONS TRACKER & ONE-TAP ADHERENCE
    # =========================================================================
    if show_meds:
        total_meds = len(meds)
        taken_meds = sum(1 for m in meds if m.get("status") == "Taken")
        adherence_pct = round((taken_meds / max(1, total_meds)) * 100.0, 1)

        with st.expander(f"💊 Active Prescriptions & Adherence Schedule ({taken_meds}/{total_meds} Taken • {adherence_pct}%)", expanded=False):
            col_hdr, col_adh = st.columns([2, 1])
            with col_hdr:
                st.caption("One-tap tracking updates your digital twin adherence record in real time.")
            with col_adh:
                adh_badge_color = "#059669" if adherence_pct >= 80 else "#dc2626"
                st.markdown(
                    f"<div style='text-align: right; font-weight: 700; color: {adh_badge_color};'>Running Adherence: {adherence_pct}%</div>",
                    unsafe_allow_html=True
                )

            for med in meds:
                m_id = med["med_id"]
                status = med.get("status", "Due")

                col_info, col_btn1, col_btn2, col_btn3 = st.columns([2.5, 1, 1, 1])
                with col_info:
                    st.markdown(
                        f"""
                        <strong>{med['med_name']}</strong> ({med['dosage']})<br>
                        <span style="color: #64748b; font-size: 0.85rem;">Schedule: {med.get('frequency', med.get('schedule_time', ''))}</span><br>
                        <span style="color: #0284c7; font-size: 0.85rem;">Instructions: {med.get('instructions', 'Take as directed')}</span>
                        """,
                        unsafe_allow_html=True
                    )

                with col_btn1:
                    if st.button("✅ Taken", key=f"taken_{m_id}", type="primary" if status == "Taken" else "secondary", use_container_width=True):
                        db.update_medication_status(m_id, "Taken", updated_by="Patient")
                        st.rerun()

                with col_btn2:
                    if st.button("⏰ Due", key=f"due_{m_id}", type="primary" if status == "Due" else "secondary", use_container_width=True):
                        db.update_medication_status(m_id, "Due", updated_by="Patient")
                        st.rerun()

                with col_btn3:
                    if st.button("❌ Missed", key=f"missed_{m_id}", type="primary" if status == "Missed" else "secondary", use_container_width=True):
                        db.update_medication_status(m_id, "Missed", updated_by="Patient")
                        st.rerun()

    # =========================================================================
    # 2. PATIENT CONTEXT HEADER & WELCOME CARD
    # =========================================================================
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%); padding: 1.3rem 1.6rem; border-radius: 12px; color: white; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="background: rgba(255,255,255,0.2); padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                        AI Healthcare Companion
                    </span>
                    <h2 style="margin: 6px 0 2px 0; font-size: 1.55rem; font-weight: 700; color: white;">
                        💬 HealthMate Assistant for {patient['full_name']}
                    </h2>
                    <p style="margin: 0; opacity: 0.92; font-size: 0.88rem;">
                        MRN: <strong>{patient['mrn']}</strong> • {patient['diagnosis_type']} ({patient['diagnosis_location']})
                    </p>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <div style="background: rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 8px; font-size: 0.78rem;">
                        👨‍⚕️ <strong>{patient.get('oncologist_name', 'Dr. Aris Thorne')}</strong>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 8px; font-size: 0.78rem;">
                        📅 Next Visit: <strong>{patient.get('next_visit_date', '28-Oct-2026')}</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Emergency Guidance Callout
    _render_emergency_callout_banner()

    # =========================================================================
    # 3. BACKGROUND INFERENCE CONFIGURATION (Silent & Automatic)
    # =========================================================================
    openai_key_in_env = os.environ.get("OPENAI_API_KEY", "")
    current_key = st.session_state.get("openai_api_key", openai_key_in_env)
    groq_key_in_env = os.environ.get("GROQ_API_KEY", "")
    current_groq_key = st.session_state.get("groq_api_key", groq_key_in_env)
    gemini_key_in_env = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    current_gemini_key = st.session_state.get("gemini_api_key", gemini_key_in_env)
    selected_model_code = "qwen/qwen3.8-27b" if current_groq_key else ("gpt-4o" if current_key else "clinical-gpt")

    # =========================================================================
    # 4. CHAT HISTORY INITIALIZATION & PRIVACY CONTROLS
    # =========================================================================
    history_key = f"companion_history_{patient['mrn']}"
    if history_key not in st.session_state:
        # Load persisted messages from database client if available
        stored_msgs = db.get_chat_history(patient_id, limit=20)
        if stored_msgs:
            st.session_state[history_key] = [
                {
                    "role": m.get("role", "assistant"),
                    "content": m.get("content", ""),
                    "model_used": m.get("model_used", "AI Health Companion"),
                    "timestamp": m.get("created_at", "")[:16].replace("T", " ")
                }
                for m in stored_msgs
            ]
        else:
            welcome_text = (
                f"Hello {first_name}! I am **HealthMate**, your personalized AI Healthcare Companion. "
                f"I am connected to your digital health twin record to help you understand your brain scans, "
                f"prescribed medications, symptoms, and daily wellness.\n\n"
                f"• Ask me about **healthy diet & nutrition** (WHO/NHS guidelines)\n"
                f"• Ask me about **common symptoms** and what signs to watch for\n"
                f"• Ask me **how to prepare for your upcoming doctor appointment**\n"
                f"• Ask me about **how your medications (Keppra & Temozolomide) work**\n\n"
                f"How can I support your health and well-being today, {first_name}?"
            )
            st.session_state[history_key] = [
                {
                    "role": "assistant",
                    "content": welcome_text,
                    "model_used": "HealthMate Companion",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
            ]

    # Quick prompt chips header & action buttons
    c_chip_hdr, c_export, c_reset = st.columns([2.5, 1, 1])
    with c_chip_hdr:
        st.markdown("##### 💡 Suggested Questions")
    with c_export:
        # Export chat transcript
        chat_transcript = f"--- HealthMate Care Companion Chat Transcript ---\nPatient: {patient['full_name']} (MRN: {patient['mrn']})\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        for m in st.session_state[history_key]:
            r_label = "Patient" if m.get("role") == "user" else "HealthMate AI"
            chat_transcript += f"[{m.get('timestamp', '')}] {r_label}:\n{m.get('content')}\n\n"
        st.download_button(
            "📥 Export Chat",
            data=chat_transcript,
            file_name=f"healthmate_chat_{patient['mrn']}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            key=f"export_chat_btn_{patient['mrn']}",
            use_container_width=True
        )
    with c_reset:
        if st.button("🗑️ Reset Chat", key=f"clear_chat_{patient['mrn']}", use_container_width=True):
            db.clear_chat_history(patient_id)
            del st.session_state[history_key]
            st.rerun()

    # Prompt chips - Row 1 (Core requested prompts)
    row1_cols = st.columns(4)
    quick_prompts_row1 = [
        "How can I maintain a healthy diet?",
        "When should I see a doctor?",
        "How can I prepare for an appointment?",
        "Explain my current condition"
    ]
    selected_chip = None
    for i, prompt in enumerate(quick_prompts_row1):
        with row1_cols[i]:
            if st.button(prompt, key=f"chip_r1_{patient['mrn']}_{i}", use_container_width=True):
                selected_chip = prompt

    # Prompt chips - Row 2 (Symptoms, medication & appointments)
    row2_cols = st.columns(4)
    quick_prompts_row2 = [
        "What are common symptoms I should watch for?",
        "What if I miss my Keppra dose?",
        "When is my next appointment?",
        "Can I drink coffee in the morning?"
    ]
    for i, prompt in enumerate(quick_prompts_row2):
        with row2_cols[i]:
            if st.button(prompt, key=f"chip_r2_{patient['mrn']}_{i}", use_container_width=True):
                selected_chip = prompt

    st.write("")

    # =========================================================================
    # 5. RENDER CONVERSATION MESSAGE BUBBLES
    # =========================================================================
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state[history_key]):
            role = msg.get("role", "assistant")
            avatar_icon = "🩺" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar_icon):
                if role == "assistant":
                    model_tag = msg.get("model_used", "AI Health Companion")
                    ts = msg.get("timestamp", "")
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 0.75rem; background: #f1f5f9; color: #475569; padding: 2px 10px; border-radius: 12px; font-weight: 600;">
                                🤖 {model_tag}
                            </span>
                            <span style="font-size: 0.72rem; color: #94a3b8;">
                                {ts}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown(_clean_content(msg.get("content", "")))

    # =========================================================================
    # 6. CHAT INPUT & RESPONSE GENERATION
    # =========================================================================
    user_input = st.chat_input("Ask anything about your health, symptoms, diet, medications, or appointments...")
    prompt_to_process = selected_chip or user_input

    if prompt_to_process:
        clean_prompt = prompt_to_process.strip()
        if not clean_prompt:
            st.warning("Please enter a valid question or message.")
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 1. Append and render user message
        st.session_state[history_key].append({
            "role": "user",
            "content": clean_prompt,
            "timestamp": now_str
        })
        with st.chat_message("user", avatar="👤"):
            st.markdown(clean_prompt)

        # 2. Synchronize environment variables for seamless inference
        if current_key:
            os.environ["OPENAI_API_KEY"] = current_key.strip()
        if current_groq_key:
            os.environ["GROQ_API_KEY"] = current_groq_key.strip()
        if current_gemini_key:
            os.environ["GEMINI_API_KEY"] = current_gemini_key.strip()
        os.environ["COMPANION_MODEL_NAME"] = selected_model_code

        # 3. Instantiate Agent & Generate Response
        agent = CareCompanionAgent()

        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Analyzing with HealthMate..."):
                try:
                    response_obj = agent.generate_response(
                        clean_prompt,
                        patient_id=patient_id,
                        conversation_history=st.session_state[history_key],
                        openai_api_key=current_key if current_key else None,
                        groq_api_key=current_groq_key if current_groq_key else None,
                        gemini_api_key=current_gemini_key if current_gemini_key else None,
                        model_name=selected_model_code,
                        save_history=True
                    )
                except Exception as ex:
                    try:
                        response_obj = agent.generate_response(
                            clean_prompt,
                            patient_id=patient_id,
                            conversation_history=st.session_state[history_key],
                            save_history=True
                        )
                    except Exception as inner_ex:
                        response_obj = {
                            "response_text": (
                                f"I apologize, but I encountered an error processing your request: {inner_ex}.\n\n"
                                f"Please try asking your question again or check with **{patient.get('nurse_name', 'your clinical navigator')}**."
                            ),
                            "model_used": "System Recovery",
                            "red_flag_detected": False,
                            "graph_rag": None
                        }

                bot_text = _clean_content(response_obj.get("response_text", ""))
                model_tag = response_obj.get("model_used", "AI Care Companion")
                rag_info = response_obj.get("graph_rag")

                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.75rem; background: #f1f5f9; color: #475569; padding: 2px 10px; border-radius: 12px; font-weight: 600;">
                            🤖 {model_tag}
                        </span>
                        <span style="font-size: 0.72rem; color: #94a3b8;">
                            {now_str}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(bot_text)

        # 4. Append to session history
        st.session_state[history_key].append({
            "role": "assistant",
            "content": bot_text,
            "model_used": model_tag,
            "timestamp": now_str,
            "graph_rag": rag_info
        })
        st.rerun()

    # Footer Disclaimer
    st.write("")
    st.caption("⚕️ HealthMate provides general educational health information and does not substitute for qualified clinical consultation. Always consult your oncology team for medical guidance.")


def render_floating_chat_widget():
    """
    Renders a compact, floating/quick-access AI Healthcare Companion drawer
    that can be embedded in any page or dashboard.
    """
    with st.expander("💬 Ask HealthMate AI (Quick Healthcare Assistant)", expanded=False):
        st.caption("Ask quick questions about symptoms, diet, appointments, or emergency guidance.")
        quick_q = st.text_input("Your health question:", placeholder="e.g., Can I drink coffee with my medicine?", key="floating_chat_input")
        if st.button("Ask Assistant", key="floating_chat_submit", type="primary"):
            if quick_q.strip():
                db = get_database_client()
                active_mrn = st.session_state.get("active_patient_mrn", "0042")
                patient = db.get_patient_by_mrn(active_mrn) or db.get_patient_by_mrn("0042")
                agent = CareCompanionAgent()
                with st.spinner("Analyzing with HealthMate..."):
                    res = agent.generate_response(quick_q, patient_id=patient["patient_id"], save_history=True)
                st.markdown(res["response_text"])
                if res.get("graph_rag"):
                    _render_graph_rag_badge(res.get("graph_rag"))
            else:
                st.warning("Please type a question before submitting.")
