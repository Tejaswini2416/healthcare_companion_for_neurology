"""
Medications & Conversational Care Companion Component (GPT-4o & Bio-LLM Architecture)
Features:
1. One-tap medication adherence tracking and live compliance percentage.
2. Full ChatGPT-like interactive Care Companion grounded in the Patient Digital Twin.
3. Model selector (OpenAI GPT-4o, GPT-4o-mini, Groq LLaMA-3, and built-in Clinical GPT).
4. Direct API key configuration card for seamless cloud LLM access.
5. Quick suggested clinical prompt chips and multi-turn conversation memory.
6. GraphRAG anti-hallucination verification badges and clinical literature citations.
"""

import os
import streamlit as st
from typing import Dict, Any, List, Optional
from ...utils.supabase_client import get_database_client
from ...models.companion_agent import CareCompanionAgent


def _render_graph_rag_badge(rag_data: Optional[Dict[str, Any]]):
    """Renders sleek GraphRAG anti-hallucination verification badge and citation drawer."""
    if not rag_data or not rag_data.get("is_graph_verified"):
        return
    score = int(rag_data.get("anti_hallucination_score", 0.95) * 100)
    citations = rag_data.get("citations", [])
    guidelines = rag_data.get("guidelines_applied", [])
    alerts = rag_data.get("clinical_alerts", [])
    nodes = rag_data.get("retrieved_nodes", [])

    st.markdown(
        f"""
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 6px 12px; margin: 8px 0; font-size: 0.8rem; color: #166534; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <strong>🛡️ GraphRAG Verified</strong> • NCCN Category 1 & AAN Standards
            </div>
            <div style="font-weight: 700; color: #15803d;">
                Anti-Hallucination Confidence: {score}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    for alert in alerts:
        st.warning(alert)

    if citations or guidelines or nodes:
        with st.expander(f"📚 Clinical Citations & Knowledge Graph Grounding ({len(citations)} citations)", expanded=False):
            if guidelines:
                st.markdown("**Applied Guidelines & Protocols**:")
                for g in guidelines:
                    st.markdown(f"- 📋 *{g}*")
            if citations:
                st.markdown("**Literature & Regulatory Evidence**:")
                for c in citations:
                    st.markdown(f"- 📖 {c}")
            if nodes:
                st.caption(f"Knowledge Graph Nodes: `{', '.join(nodes)}`")


def render_companion_chat(show_meds: bool = True):
    """Renders the Medications Tracker and Conversational Care Companion GPT."""
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

    # =========================================================================
    # 1. MEDICATIONS TRACKER & ONE-TAP ADHERENCE (Optional)
    # =========================================================================
    if show_meds:
        st.subheader(f"💊 Prescriptions & Adherence Schedule for {patient['full_name']}")

        total_meds = len(meds)
        taken_meds = sum(1 for m in meds if m.get("status") == "Taken")
        adherence_pct = round((taken_meds / max(1, total_meds)) * 100.0, 1)

        col_hdr, col_adh = st.columns([2, 1])
        with col_hdr:
            st.caption("Tap any button below to update your adherence status in real time.")
        with col_adh:
            adh_badge_color = "#059669" if adherence_pct >= 80 else "#dc2626"
            st.markdown(
                f"<div style='text-align: right; font-weight: 700; color: {adh_badge_color};'>Running Adherence: {adherence_pct}% ({taken_meds}/{total_meds} Taken)</div>",
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

        st.divider()

    # =========================================================================
    # 2. CONVERSATIONAL CARE COMPANION (CHATGPT-STYLE AI CHATBOT)
    # =========================================================================
    st.subheader("💬 AI Care Companion")
    st.caption(
        "Chat naturally with your AI health companion. "
        "Directly grounded in your 3D brain scans, medication history, and evidence-based clinical guidelines."
    )

    # Model Configuration & API Key Expander
    openai_key_in_env = os.environ.get("OPENAI_API_KEY", "")
    current_key = st.session_state.get("openai_api_key", openai_key_in_env)
    groq_key_in_env = os.environ.get("GROQ_API_KEY", "")
    current_groq_key = st.session_state.get("groq_api_key", groq_key_in_env)

    # Top Toolbar: Model Selector & Live Status
    col_mod, col_stat = st.columns([1.5, 1.2])

    with col_mod:
        model_display_names = {
            "OpenAI GPT-4o (State-of-the-Art)": "gpt-4o",
            "OpenAI GPT-4o Mini (Fast & Cost-Effective)": "gpt-4o-mini",
            "Groq LLaMA-3.3 70B (High-Speed Cloud LLM)": "llama-3.3-70b-versatile",
            "Digital Twin Clinical GPT (Built-in)": "clinical-gpt"
        }
        selected_display = st.selectbox(
            "🧠 Chatbot LLM Engine:",
            options=list(model_display_names.keys()),
            index=0 if current_key else 3,
            key=f"llm_model_select_{patient['mrn']}"
        )
        selected_model_code = model_display_names[selected_display]

    with col_stat:
        if current_key and "gpt" in selected_model_code:
            st.markdown(
                """
                <div style="background: #f0fdf4; border: 1.5px solid #22c55e; border-radius: 8px; padding: 10px 14px; margin-top: 18px;">
                    <span style="color: #15803d; font-weight: 700; font-size: 0.85rem;">🟢 OpenAI GPT Connected</span><br>
                    <span style="color: #475569; font-size: 0.75rem;">Direct cloud inference with Digital Twin grounding</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif current_groq_key and "llama" in selected_model_code:
            st.markdown(
                """
                <div style="background: #eff6ff; border: 1.5px solid #3b82f6; border-radius: 8px; padding: 10px 14px; margin-top: 18px;">
                    <span style="color: #1d4ed8; font-weight: 700; font-size: 0.85rem;">🚀 Groq Cloud Active</span><br>
                    <span style="color: #475569; font-size: 0.75rem;">High-speed 70B parameter inference</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 10px 14px; margin-top: 18px;">
                    <span style="color: #0f172a; font-weight: 700; font-size: 0.85rem;">⚡ Built-in Clinical GPT Active</span><br>
                    <span style="color: #64748b; font-size: 0.75rem;">Direct AI inference with Digital Twin grounding</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Initialize chat history in session state per patient
    history_key = f"companion_history_{patient['mrn']}"
    first_name = patient['full_name'].split()[0]
    if history_key not in st.session_state:
        st.session_state[history_key] = [
            {
                "role": "assistant",
                "content": (
                    f"Hello {first_name}! I am your AI Care Companion. "
                    "I am directly connected to your digital health twin record to help you understand your scans, "
                    "medications, symptoms, and daily wellness. How can I help you today?"
                ),
                "model_used": "AI Health Companion"
            }
        ]
    else:
        # Auto-upgrade any legacy robotic canned messages from older sessions
        for msg in st.session_state[history_key]:
            c = msg.get("content", "")
            if msg.get("role") == "assistant" and any(k in c for k in ["Regarding your question:", "Care Companion Response for", "Clinical Insights from Your Digital Health Twin:", "### 🌟 Your Brain Story"]):
                from ...models import companion_agent
                _upgrade_agent = companion_agent.CareCompanionAgent()
                msg["content"] = _upgrade_agent._generate_clinical_gpt_response(
                    "Explain my diagnosis simply",
                    patient=patient,
                    scans=db.get_patient_scans(patient_id),
                    meds=db.get_patient_medications(patient_id),
                    symptoms=db.get_patient_symptoms(patient_id),
                    timeline=db.get_patient_timeline(patient_id)
                )

    # Quick prompt chips and clear conversation button
    c_chip_hdr, c_reset = st.columns([3, 1])
    with c_chip_hdr:
        st.markdown("##### 💡 Suggested Questions")
    with c_reset:
        if st.button("🗑️ Reset Chat", key=f"clear_chat_{patient['mrn']}", use_container_width=True):
            del st.session_state[history_key]
            st.rerun()

    row1_cols = st.columns(4)
    quick_prompts_row1 = [
        "Explain my current condition",
        "Explain my diagnosis simply",
        "Explain my latest scan results",
        "Can I drink coffee in the morning?"
    ]
    selected_chip = None
    for i, prompt in enumerate(quick_prompts_row1):
        with row1_cols[i]:
            if st.button(prompt, key=f"chip_r1_{patient['mrn']}_{i}", use_container_width=True):
                selected_chip = prompt

    row2_cols = st.columns(4)
    quick_prompts_row2 = [
        "What if I vomit my Temozolomide?",
        "What diet is best for brain recovery?",
        "Can I drive a car right now?",
        "What should I ask Dr. Thorne?"
    ]
    for i, prompt in enumerate(quick_prompts_row2):
        with row2_cols[i]:
            if st.button(prompt, key=f"chip_r2_{patient['mrn']}_{i}", use_container_width=True):
                selected_chip = prompt

    # Render previous messages
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "👤"):
            if msg["role"] == "assistant" and msg.get("model_used"):
                st.markdown(
                    f"<span style='font-size:0.75rem; background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:12px; font-weight:600; display:inline-block; margin-bottom:6px;'>🤖 {msg['model_used']}</span>",
                    unsafe_allow_html=True
                )
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about your diagnosis, scan results, medications, or daily life...")
    prompt_to_process = selected_chip or user_input

    if prompt_to_process:
        # Display user message
        st.session_state[history_key].append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_to_process)

        # Synchronize environment variables for seamless inference
        if current_key:
            os.environ["OPENAI_API_KEY"] = current_key.strip()
        if current_groq_key:
            os.environ["GROQ_API_KEY"] = current_groq_key.strip()
        os.environ["COMPANION_MODEL_NAME"] = selected_model_code

        # Generate response from twin-grounded agent
        import importlib
        from ...models import companion_agent
        importlib.reload(companion_agent)
        agent = companion_agent.CareCompanionAgent()

        with st.chat_message("assistant", avatar="🩺"):
            spinner_text = "Thinking (OpenAI GPT-4o)..." if (current_key and "gpt" in selected_model_code) else "Thinking..."
            with st.spinner(spinner_text):
                try:
                    response_obj = agent.generate_response(
                        prompt_to_process,
                        patient_id=patient_id,
                        conversation_history=st.session_state[history_key],
                        openai_api_key=current_key if current_key else None,
                        model_name=selected_model_code,
                        groq_api_key=current_groq_key if current_groq_key else None
                    )
                except Exception:
                    response_obj = agent.generate_response(
                        prompt_to_process,
                        patient_id=patient_id,
                        conversation_history=st.session_state[history_key]
                    )
                bot_text = response_obj["response_text"]
                model_tag = response_obj.get("model_used", "AI Care Companion")

                st.markdown(
                    f"<span style='font-size:0.75rem; background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:12px; font-weight:600; display:inline-block; margin-bottom:6px;'>🤖 {model_tag}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(bot_text)

        st.session_state[history_key].append({
            "role": "assistant",
            "content": bot_text,
            "model_used": model_tag,
            "cross_modal_insight": response_obj.get("cross_modal_insight")
        })
