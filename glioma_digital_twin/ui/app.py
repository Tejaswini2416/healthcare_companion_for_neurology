"""
Main Streamlit Application: Healthcare Companion
AI-Powered Cancer Care Companion (Multimodal Learning, Biophysical In-Silico Simulation & Patient Health Twin for Brain Cancer)
Integrates Dual-Role Clinical Workstation and Doctor-Governed Patient Portal.
"""

import os
import sys

# Neutralize PyTorch torch.classes inspection conflict with Streamlit local_sources_watcher
try:
    import torch
    if hasattr(torch, "_classes") and hasattr(torch._classes, "_Classes"):
        _orig_classes_getattr = torch._classes._Classes.__getattr__
        def _safe_classes_getattr(self, attr):
            if attr in ("__path__", "_path", "__file__", "__loader__") or attr.startswith("__"):
                raise AttributeError(f"'_Classes' object has no attribute '{attr}'")
            return _orig_classes_getattr(self, attr)
        torch._classes._Classes.__getattr__ = _safe_classes_getattr
except Exception:
    pass

import streamlit as st

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from glioma_digital_twin.ui.components.patient_dashboard import render_patient_dashboard
from glioma_digital_twin.ui.components.symptom_logger import render_symptom_logger
from glioma_digital_twin.ui.components.report_viewer import render_report_viewer
from glioma_digital_twin.ui.components.companion_chat import render_companion_chat
from glioma_digital_twin.ui.components.longitudinal_trends import render_longitudinal_trends
from glioma_digital_twin.ui.components.clinician_upload import render_clinician_workstation
from glioma_digital_twin.ui.components.mri_evaluator import render_mri_evaluator
from glioma_digital_twin.ui.components.doctor_login import (
    render_doctor_login,
    is_doctor_authenticated,
    get_current_doctor,
    logout_doctor
)
from glioma_digital_twin.utils.supabase_client import get_database_client


def main():
    st.set_page_config(
        page_title="Healthcare Companion: AI Cancer Care",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #f8fafc;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
        }
        @media print {
            header, footer, .stSidebar, .stButton, div[data-testid="stToolbar"] {
                display: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session states - Entry page is strictly the Doctor Login page
    if "current_view" not in st.session_state:
        st.session_state.current_view = "login"
    if "active_patient_mrn" not in st.session_state:
        st.session_state.active_patient_mrn = "0042"

    def navigate_to(view_name: str):
        st.session_state.current_view = view_name
        st.rerun()

    db = get_database_client()
    authenticated = is_doctor_authenticated()
    current_doc = get_current_doctor()

    # Security Guard: Without logging in as doctor, patient information must not be disclosed
    if not authenticated:
        st.session_state.current_view = "login"

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/brain.png", width=64)
        st.title("Healthcare Companion")
        st.caption("AI-Powered Cancer Care Companion & Biophysical In-Silico Patient Twin")

        # Note: Supabase settings badge removed per user request

        if not authenticated:
            st.divider()
            st.markdown(
                """
                <div style="background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-top: 10px;">
                    <div style="font-weight: 700; font-size: 0.88rem; color: #1e293b; display: flex; align-items: center; gap: 6px;">
                        🔒 Doctor Sign-In Required
                    </div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 6px; line-height: 1.4;">
                        Patient health information is protected. Please sign in as an attending physician on the main page to unlock patient records and the clinical workstation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.divider()
            # Physician Profile Card
            st.markdown(
                f"""
                <div style="background: #0f766e15; border: 1.5px solid #0f766e; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <span style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #0f766e;">Physician On Service</span><br>
                    <strong style="color: #0f172a; font-size: 0.95rem;">{current_doc['avatar']} {current_doc['full_name']}</strong><br>
                    <span style="font-size: 0.78rem; color: #475569;">{current_doc['role']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("🔒 Sign Out Physician", key="sidebar_logout_btn", use_container_width=True):
                logout_doctor()

            st.divider()

            # Active Patient Selector (Unlocked only after Doctor Login)
            patients = db.get_all_patients()
            assigned_mrns = current_doc.get("assigned_mrns", [p["mrn"] for p in patients])
            assigned_patients = [p for p in patients if p["mrn"] in assigned_mrns] or patients

            st.markdown("##### 👤 Active Patient Context")
            patient_options = {
                f"{p['full_name']} (MRN: {p['mrn']})": p['mrn']
                for p in assigned_patients
            }
            # Also append any other cohort members
            for p in patients:
                label = f"{p['full_name']} (MRN: {p['mrn']})"
                if label not in patient_options:
                    patient_options[label] = p['mrn']

            current_mrn = st.session_state.active_patient_mrn
            cur_idx = 0
            for i, (label, mrn) in enumerate(patient_options.items()):
                if mrn == current_mrn:
                    cur_idx = i
                    break

            selected_patient_str = st.selectbox(
                "Select Active Patient:",
                options=list(patient_options.keys()),
                index=cur_idx,
                key="global_sidebar_patient_selector"
            )
            new_mrn = patient_options[selected_patient_str]
            if new_mrn != st.session_state.active_patient_mrn:
                st.session_state.active_patient_mrn = new_mrn
                st.rerun()

            active_patient = db.get_patient_by_mrn(st.session_state.active_patient_mrn) or patients[0]

            st.divider()
            st.markdown("##### 📍 Clinical Navigation")

            nav_items = [
                ("👨‍⚕️ Doctor Cohort Inspector", "login"),
                ("🩺 Clinician Workstation", "clinician"),
                ("🏠 Patient Twin Dashboard", "dashboard"),
                ("📤 Upload & Evaluate Scan", "evaluator"),
                ("📋 Scans & Translated Reports", "reports"),
                ("📝 Log Symptoms & Triage", "symptoms"),
                ("💊 Medications & Care Companion", "companion"),
                ("📈 Longitudinal Trends", "trends"),
            ]

            for label, view in nav_items:
                btn_type = "primary" if st.session_state.current_view == view else "secondary"
                if st.button(label, key=f"nav_{view}", type=btn_type, use_container_width=True):
                    st.session_state.current_view = view
                    st.rerun()

            st.divider()
            st.caption(
                f"Active Patient: **{active_patient['full_name']}** (`{active_patient['mrn']}`)\n\n"
                f"{active_patient['diagnosis_type']}\n\n"
                f"Location: {active_patient['diagnosis_location']}"
            )

    # --- Main Content Rendering ---
    if not authenticated:
        # Strictly render Doctor Login as the entry page
        render_doctor_login(on_success=lambda: navigate_to("login"))
    else:
        # Doctor is authenticated: Route to the requested view
        if st.session_state.current_view == "login":
            render_doctor_login(on_success=lambda: navigate_to("clinician"))
        elif st.session_state.current_view == "clinician":
            render_clinician_workstation()
        elif st.session_state.current_view == "dashboard":
            render_patient_dashboard(on_navigate=navigate_to)
        elif st.session_state.current_view == "evaluator":
            render_mri_evaluator()
        elif st.session_state.current_view == "reports":
            render_report_viewer()
        elif st.session_state.current_view == "symptoms":
            render_symptom_logger()
        elif st.session_state.current_view == "companion" or st.session_state.current_view == "medications":
            render_companion_chat()
        elif st.session_state.current_view == "trends":
            render_longitudinal_trends()
        else:
            render_doctor_login(on_success=lambda: navigate_to("clinician"))


if __name__ == "__main__":
    main()
