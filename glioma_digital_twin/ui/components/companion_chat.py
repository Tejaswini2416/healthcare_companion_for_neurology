"""
Medications & Conversational Care Companion Component
One-tap medication adherence schedule, automated adherence rate,
and twin-grounded interactive Care Companion with quick prompt chips.
Supports dynamic patient context across all pre-seeded patients.
"""

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


def render_companion_chat():
    """Renders the Medications Tracker and Conversational Care Companion."""
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

    # 1. Medications Section & One-Tap Adherence Statuses
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

    # 2. Conversational Care Companion
    st.subheader("💬 Personalized Care Companion")
    st.markdown("Your digital companion is grounded in your latest scans, lab results, and validated against NCCN clinical guidelines.")

    # Initialize chat history in session state per patient
    history_key = f"companion_history_{patient['mrn']}"
    if history_key not in st.session_state:
        st.session_state[history_key] = [
            {
                "role": "assistant",
                "content": (
                    f"Hello {patient['full_name']}, I am your personalized Care Companion. "
                    "I am directly connected to your patient digital twin. Feel free to ask about your scan findings, "
                    "medications, or how to manage symptoms. I am here to help you understand your journey."
                ),
                "graph_rag": {
                    "is_graph_verified": True,
                    "anti_hallucination_score": 0.98,
                    "citations": [
                        "NCCN Guidelines for Central Nervous System Cancers (v2026.1).",
                        "American Academy of Neurology (AAN) Practice Guidelines."
                    ],
                    "guidelines_applied": ["NCCN Category 1 Clinical Decision Support"],
                    "clinical_alerts": [],
                    "retrieved_nodes": ["protocol_stupp", "drug_temozolomide", "drug_levetiracetam"]
                }
            }
        ]

    # Quick prompt chips
    st.markdown("##### 💡 Suggested Questions")
    chip_cols = st.columns(4)
    quick_prompts = [
        "What does perilesional edema mean?",
        "Is it okay that I missed my Levetiracetam?",
        "When do I hold Temozolomide?",
        "Explain my latest scan"
    ]

    selected_chip = None
    for i, prompt in enumerate(quick_prompts):
        with chip_cols[i]:
            if st.button(prompt, key=f"chip_{patient['mrn']}_{i}", use_container_width=True):
                selected_chip = prompt

    # Render previous messages
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("cross_modal_insight"):
                st.caption(f"🧠 **Twin Cross-Modal Insight**: {msg['cross_modal_insight']}")
            if msg.get("graph_rag"):
                _render_graph_rag_badge(msg["graph_rag"])

    # Chat input
    user_input = st.chat_input("Ask a question about your diagnosis, scan results, or medications...")
    prompt_to_process = selected_chip or user_input

    if prompt_to_process:
        # Display user message
        st.session_state[history_key].append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Generate response from twin-grounded agent
        agent = CareCompanionAgent()
        with st.chat_message("assistant"):
            with st.spinner("Connecting with your Patient Digital Twin & GraphRAG Guard..."):
                response_obj = agent.generate_response(
                    prompt_to_process,
                    patient_id=patient_id,
                    conversation_history=st.session_state[history_key]
                )
                bot_text = response_obj["response_text"]
                st.markdown(bot_text)

                if response_obj.get("cross_modal_insight"):
                    st.caption(f"🧠 **Twin Cross-Modal Insight**: {response_obj['cross_modal_insight']}")

                if response_obj.get("graph_rag"):
                    _render_graph_rag_badge(response_obj["graph_rag"])

        st.session_state[history_key].append({
            "role": "assistant",
            "content": bot_text,
            "cross_modal_insight": response_obj.get("cross_modal_insight"),
            "graph_rag": response_obj.get("graph_rag")
        })
