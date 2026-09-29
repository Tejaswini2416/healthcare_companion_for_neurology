"""
Main Streamlit Application: Healthcare Companion
AI-Powered Cancer Care Companion (Multimodal Learning, Biophysical In-Silico Simulation & Patient Health Twin for Brain Cancer)
Integrates Dual-Role Clinical Workstation and Patient Portal across pre-seeded cohorts.
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

    # Initialize session states
    if "current_view" not in st.session_state:
        st.session_state.current_view = "dashboard"
    if "active_patient_mrn" not in st.session_state:
        st.session_state.active_patient_mrn = "0042"

    def navigate_to(view_name: str):
        st.session_state.current_view = view_name
        st.rerun()

    db = get_database_client()
    patients = db.get_all_patients()

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/brain.png", width=64)
        st.title("Healthcare Companion")
        st.caption("AI-Powered Cancer Care Companion & Biophysical In-Silico Patient Twin")

        st.divider()

        # Global Active Patient Context Selector
        st.markdown("##### 👤 Active Patient Context")
        patient_options = {
            f"{p['full_name']} (MRN: {p['mrn']})": p['mrn']
            for p in patients
        }
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

        # Role switcher
        user_role = st.selectbox(
            "User Journey Portal",
            options=[
                f"Patient Portal ({active_patient['full_name']})",
                "Clinician Workstation",
                "📤 Upload & Evaluate MRI Scan"
            ],
            index=0,
            key="portal_role_switcher"
        )

        st.divider()

        if user_role.startswith("Patient Portal"):
            st.markdown(f"##### 📍 Patient Navigation")
            nav_items = [
                ("🏠 Home Dashboard", "dashboard"),
                ("📤 Upload & Evaluate Scan", "evaluator"),
                ("📋 Scans & Translated Reports", "reports"),
                ("📝 Log Symptoms", "symptoms"),
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
                f"Active: **{active_patient['full_name']}** (MRN: `{active_patient['mrn']}`)\n"
                f"{active_patient['diagnosis_type']}\n"
                f"Stage: {active_patient['treatment_stage']}"
            )

        elif user_role == "Clinician Workstation":
            st.session_state.current_view = "clinician"
            st.markdown("##### 🩺 Clinical Station")
            st.caption("Neuro-Oncology Console: SegResNet 3D segmentation, RANO 2.0 triage, prescription governance, and In-Silico Horizon modeling.")

        elif user_role == "📤 Upload & Evaluate MRI Scan":
            st.session_state.current_view = "evaluator"
            st.markdown("##### 📤 Diagnostic Imaging Station")
            st.caption("Real-time Brain MRI scan upload, tumor detection, and Grad-CAM explainability.")

    # --- Main Content Rendering ---
    if st.session_state.current_view == "dashboard":
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
    elif st.session_state.current_view == "clinician":
        render_clinician_workstation()
    else:
        render_patient_dashboard(on_navigate=navigate_to)


if __name__ == "__main__":
    main()
