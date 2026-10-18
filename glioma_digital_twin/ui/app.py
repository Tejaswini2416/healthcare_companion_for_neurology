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
from glioma_digital_twin.ui.components.report_verification_hub import render_report_verification_hub
from glioma_digital_twin.ui.components.doctor_report_writer import render_report_writing_page
from glioma_digital_twin.ui.components.brain_3d_viewer import render_3d_brain_viewer_panel
from glioma_digital_twin.ui.components.doctor_login import (
    render_doctor_login,
    is_doctor_authenticated,
    get_current_doctor,
    logout_doctor,
    is_patient_authenticated,
    get_current_patient,
    logout_patient,
    is_user_authenticated,
    logout_any_user
)
from glioma_digital_twin.utils.supabase_client import get_database_client


def main():
    st.set_page_config(
        page_title="AI-Powered Cancer Care Companion",
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

    # Initialize session states - Entry page is the Auth Portal
    if "current_view" not in st.session_state:
        st.session_state.current_view = "login"
    if "active_patient_mrn" not in st.session_state:
        st.session_state.active_patient_mrn = "0042"

    def navigate_to(view_name: str):
        st.session_state.current_view = view_name
        st.rerun()

    db = get_database_client()
    doc_authenticated = is_doctor_authenticated()
    pat_authenticated = is_patient_authenticated()
    user_authenticated = doc_authenticated or pat_authenticated
    current_doc = get_current_doctor()
    current_patient = get_current_patient()

    # If patient is logged in, sync their MRN with the active session MRN
    if pat_authenticated and current_patient:
        st.session_state.active_patient_mrn = current_patient["mrn"]

    # Security Guard: Without logging in, portal remains on login
    if not user_authenticated:
        st.session_state.current_view = "login"

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/brain.png", width=64)
        st.title("AI-Powered Cancer Care Companion")

        if not user_authenticated:
            st.divider()
            st.markdown(
                """
                <div style="background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-top: 10px;">
                    <div style="font-weight: 700; font-size: 0.88rem; color: #1e293b; display: flex; align-items: center; gap: 6px;">
                        🔒 Sign-In Required
                    </div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 6px; line-height: 1.4;">
                        Please sign in as an attending physician or as a registered patient on the main portal to access records and tools.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif pat_authenticated and current_patient:
            st.divider()
            # Patient Profile Card in Sidebar
            st.markdown(
                f"""
                <div style="background: #ecfdf5; border: 1.5px solid #10b981; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <span style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #047857;">Logged In Patient</span><br>
                    <strong style="color: #064e3b; font-size: 0.95rem;">👤 {current_patient.get('full_name', 'Patient')}</strong><br>
                    <span style="font-size: 0.78rem; color: #047857;">MRN: {current_patient.get('mrn')} • {current_patient.get('age', '--')} yrs</span><br>
                    <span style="font-size: 0.74rem; color: #374151;">{current_patient.get('diagnosis_type', 'Glioma')}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("🔒 Sign Out Patient", key="sidebar_patient_logout_btn", use_container_width=True):
                logout_patient()

            st.divider()
            st.markdown("##### 📍 Patient Navigation")
            patient_nav_items = [
                ("🏠 My Health Dashboard", "dashboard"),
                ("💬 Care Companion & Questions", "companion"),
                ("📝 Log My Symptoms", "symptoms"),
                ("📋 My Scans & Reports", "reports"),
                ("🧠 3D Brain & Tumor Localization", "mri_3d"),
                ("📈 My Health Trends", "trends"),
                ("📤 Upload & Scan MRI", "evaluator"),
                ("📄 Upload Reports", "report_verify"),
                ("👤 Portal & Profile", "login"),
            ]
            for label, view in patient_nav_items:
                btn_type = "primary" if st.session_state.current_view == view else "secondary"
                if st.button(label, key=f"pat_nav_{view}", type=btn_type, use_container_width=True):
                    st.session_state.current_view = view
                    st.rerun()

            st.divider()
            st.caption(
                f"Active Patient: **{current_patient['full_name']}** (`{current_patient['mrn']}`)\n\n"
                f"{current_patient.get('diagnosis_type', 'Glioma')}\n\n"
                f"Location: {current_patient.get('diagnosis_location', 'Brain')}\n\n"
                f"Oncologist: {current_patient.get('oncologist_name', 'Dr. Aris Thorne, MD')}"
            )
        elif doc_authenticated and current_doc:
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

            options_list = list(patient_options.keys())
            if st.session_state.get("last_synced_sidebar_mrn") != current_mrn and cur_idx < len(options_list):
                st.session_state["global_sidebar_patient_selector"] = options_list[cur_idx]
                st.session_state["last_synced_sidebar_mrn"] = current_mrn

            selected_patient_str = st.selectbox(
                "Select Active Patient:",
                options=options_list,
                index=cur_idx,
                key="global_sidebar_patient_selector"
            )
            new_mrn = patient_options.get(selected_patient_str, current_mrn)
            if new_mrn != st.session_state.active_patient_mrn:
                st.session_state.active_patient_mrn = new_mrn
                st.session_state["last_synced_sidebar_mrn"] = new_mrn
                st.rerun()

            active_patient = db.get_patient_by_mrn(st.session_state.active_patient_mrn) or patients[0]

            st.divider()
            st.markdown("##### 📍 Clinical Navigation")

            nav_items = [
                ("👨‍⚕️ Doctor Cohort Inspector", "login"),
                ("🩺 Clinician Workstation", "clinician"),
                ("📝 Write & Log Reports", "write_report"),
                ("🧠 3D Brain & MRI Volumetrics", "mri_3d"),
                ("💊 Medication Governance", "medications"),
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
    def handle_auth_success():
        if is_patient_authenticated():
            navigate_to("dashboard")
        else:
            navigate_to("clinician")

    if not user_authenticated:
        # Strictly render Doctor & Patient Login Portal as the entry page
        render_doctor_login(on_success=handle_auth_success)
    else:
        # Route to the requested view
        if st.session_state.current_view == "login":
            render_doctor_login(on_success=handle_auth_success)
        elif st.session_state.current_view == "clinician":
            if pat_authenticated:
                render_patient_dashboard(on_navigate=navigate_to)
            else:
                render_clinician_workstation()
        elif st.session_state.current_view == "write_report":
            if pat_authenticated:
                render_patient_dashboard(on_navigate=navigate_to)
            else:
                render_report_writing_page()
        elif st.session_state.current_view == "mri_3d":
            db = get_database_client()
            active_mrn = st.session_state.get("active_patient_mrn", "0042")
            patient = db.get_patient_by_mrn(active_mrn) or db.get_patient_by_mrn("0042")
            scans = db.get_patient_scans(patient["patient_id"]) if patient else []
            latest_scan = scans[-1] if scans else {}
            render_3d_brain_viewer_panel(
                patient=patient,
                scan_data=latest_scan,
                panel_title=f"🧠 3D Brain Volumetric Reconstruction: {patient['full_name']} (MRN: {patient['mrn']})",
                key_prefix="global_app_3d_brain"
            )
        elif st.session_state.current_view == "dashboard":
            if doc_authenticated and not pat_authenticated:
                render_clinician_workstation()
            else:
                render_patient_dashboard(on_navigate=navigate_to)
        elif st.session_state.current_view == "evaluator":
            if doc_authenticated and not pat_authenticated:
                render_clinician_workstation()
            else:
                render_mri_evaluator()
        elif st.session_state.current_view == "reports":
            if doc_authenticated and not pat_authenticated:
                render_report_writing_page()
            else:
                render_report_viewer()
        elif st.session_state.current_view == "report_verify":
            if doc_authenticated and not pat_authenticated:
                render_report_writing_page()
            else:
                db = get_database_client()
                active_mrn = st.session_state.get("active_patient_mrn", "0042")
                patient = db.get_patient_by_mrn(active_mrn) or db.get_patient_by_mrn("0042")
                scans = db.get_patient_scans(patient["patient_id"]) if patient else []
                latest_scan = scans[-1] if scans else {}
                molecular = db.get_patient_molecular(patient["patient_id"]) if patient else {}
                render_report_verification_hub(
                    patient=patient,
                    db=db,
                    latest_scan=latest_scan,
                    molecular=molecular,
                    is_patient_mode=True
                )
        elif st.session_state.current_view == "symptoms":
            if doc_authenticated and not pat_authenticated:
                render_clinician_workstation()
            else:
                render_symptom_logger()
        elif st.session_state.current_view == "companion":
            if doc_authenticated and not pat_authenticated:
                # Companion chatbot removed from Doctor portal
                render_report_writing_page()
            else:
                render_companion_chat()
        elif st.session_state.current_view == "medications":
            if doc_authenticated and not pat_authenticated:
                render_clinician_workstation()
            else:
                render_companion_chat()
        elif st.session_state.current_view == "trends":
            render_longitudinal_trends()
        else:
            if pat_authenticated:
                render_patient_dashboard(on_navigate=navigate_to)
            else:
                render_clinician_workstation()

        # Active HealthMate AI Care Companion Dock on every patient sub-page
        if pat_authenticated and st.session_state.current_view not in ["companion", "dashboard", "login"]:
            st.write("")
            st.divider()
            with st.expander("💬 Ask HealthMate (AI Cancer Care Companion)", expanded=False):
                render_companion_chat(show_meds=False)


if __name__ == "__main__":
    main()
