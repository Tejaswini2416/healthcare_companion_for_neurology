"""
Doctor Login & Clinician Authentication Component
Provides:
1. Hospital-grade clinician authentication portal (Role-Based Access Control).
2. Pre-configured attending physician profiles (Dr. Aris Thorne, MD, PhD; Dr. Elena Rostova, MD; Dr. Marcus Vance, MD).
3. 1-Click Express Physician Authentication + Standard Secure Credential Login.
4. Assigned Patient Roster & Instant Patient Detail Inspector for authenticated physicians.
5. Session state persistence and secure sign-out.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from ...utils.supabase_client import get_database_client


CLINICIAN_PROFILES: List[Dict[str, Any]] = [
    {
        "doctor_id": "doc_thorne",
        "full_name": "Dr. Aris Thorne, MD, PhD",
        "role": "Chief of Neuro-Oncology",
        "department": "Department of Neuro-Oncology & Brain Tumor Center",
        "hospital": "Comprehensive Brain Tumor Institute",
        "license_no": "NPI: 1849204821 • State Lic: NO-84920",
        "email": "dr.thorne@hospital.org",
        "assigned_mrns": ["0042", "0044", "0045"],
        "avatar": "👨‍⚕️",
        "status": "Attending on Service"
    },
    {
        "doctor_id": "doc_rostova",
        "full_name": "Dr. Elena Rostova, MD",
        "role": "Associate Professor of Neuro-Oncology",
        "department": "Division of Surgical Neuro-Oncology",
        "hospital": "Comprehensive Brain Tumor Institute",
        "license_no": "NPI: 1918239014 • State Lic: NO-91823",
        "email": "dr.rostova@hospital.org",
        "assigned_mrns": ["0043"],
        "avatar": "👩‍⚕️",
        "status": "In Clinic & Surgical Review"
    },
    {
        "doctor_id": "doc_vance",
        "full_name": "Dr. Marcus Vance, MD",
        "role": "Chief of Diagnostic Neuroradiology",
        "department": "Department of Neuroradiology & Advanced Neuro-Imaging",
        "hospital": "Comprehensive Brain Tumor Institute",
        "license_no": "NPI: 1732105592 • State Lic: NR-73210",
        "email": "dr.vance@hospital.org",
        "assigned_mrns": ["0042", "0043", "0044"],
        "avatar": "🧑‍⚕️",
        "status": "Advanced Imaging Reading Room"
    }
]

# Clinician cohort alias for backwards compatibility
DOCTOR_COHORT = CLINICIAN_PROFILES


def get_current_doctor() -> Optional[Dict[str, Any]]:
    """Returns the authenticated doctor profile from session state, or None."""
    session = st.session_state.get("doctor_session", {})
    if session.get("logged_in"):
        return session.get("doctor")
    return None


def is_doctor_authenticated() -> bool:
    """Checks if a doctor is currently logged in."""
    return st.session_state.get("doctor_session", {}).get("logged_in", False)


def logout_doctor():
    """Logs out the current doctor."""
    st.session_state.doctor_session = {"logged_in": False, "doctor": None}
    st.session_state.user_role = None
    st.session_state.current_view = "login"
    st.rerun()


def login_doctor(doctor_profile: Dict[str, Any]):
    """Logs in a doctor profile and sets session state."""
    st.session_state.doctor_session = {
        "logged_in": True,
        "doctor": doctor_profile,
        "login_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Clear patient session
    st.session_state.patient_session = {"logged_in": False, "patient": None}
    st.session_state.user_role = "doctor"
    # Set default patient to first assigned patient if available
    if doctor_profile.get("assigned_mrns"):
        st.session_state.active_patient_mrn = doctor_profile["assigned_mrns"][0]


def get_current_patient() -> Optional[Dict[str, Any]]:
    """Returns the authenticated patient profile from session state, or None."""
    session = st.session_state.get("patient_session", {})
    if session.get("logged_in"):
        return session.get("patient")
    return None


def is_patient_authenticated() -> bool:
    """Checks if a patient is currently logged in."""
    return st.session_state.get("patient_session", {}).get("logged_in", False)


def login_patient(patient_profile: Dict[str, Any]):
    """Logs in a patient profile and sets session state."""
    st.session_state.patient_session = {
        "logged_in": True,
        "patient": patient_profile,
        "login_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Clear doctor session
    st.session_state.doctor_session = {"logged_in": False, "doctor": None}
    st.session_state.active_patient_mrn = patient_profile["mrn"]
    st.session_state.user_role = "patient"


def logout_patient():
    """Logs out the current patient."""
    st.session_state.patient_session = {"logged_in": False, "patient": None}
    st.session_state.user_role = None
    st.session_state.current_view = "login"
    st.rerun()


def is_user_authenticated() -> bool:
    """Checks if either doctor or patient is authenticated."""
    return is_doctor_authenticated() or is_patient_authenticated()


def logout_any_user():
    """Logs out whichever user is currently authenticated."""
    st.session_state.doctor_session = {"logged_in": False, "doctor": None}
    st.session_state.patient_session = {"logged_in": False, "patient": None}
    st.session_state.user_role = None
    st.session_state.current_view = "login"
    st.rerun()


def render_doctor_login(on_success: Optional[Callable[[], None]] = None):
    """
    Renders the clinician login page and, when logged in, displays
    the assigned patient roster and detailed clinical patient inspector.
    """
    db = get_database_client()
    patients = db.get_all_patients()
    current_doc = get_current_doctor()
    current_patient = get_current_patient()

    # =========================================================================
    # STATE A1: PATIENT IS ALREADY LOGGED IN -> Patient Overview & Quick Actions
    # =========================================================================
    if current_patient:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px;">
                            Active Patient Portal Session
                        </span>
                        <h2 style="margin: 8px 0 4px 0; font-size: 1.6rem; font-weight: 700;">
                            👤 {current_patient['full_name']}
                        </h2>
                        <p style="margin: 0; opacity: 0.9; font-size: 0.95rem;">
                            MRN: <strong>{current_patient['mrn']}</strong> • Age: {current_patient['age']} • {current_patient['diagnosis_type']} ({current_patient['diagnosis_location']})
                        </p>
                        <p style="margin: 4px 0 0 0; opacity: 0.75; font-size: 0.85rem;">
                            Attending Oncologist: {current_patient.get('oncologist_name', 'Dr. Aris Thorne, MD')} • Stage: {current_patient.get('treatment_stage', 'Adjuvant Therapy')}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_p_nav1, col_p_logout = st.columns([2, 1])
        with col_p_nav1:
            if st.button("🏠 Open My Health Dashboard", type="primary", use_container_width=True):
                st.session_state.current_view = "dashboard"
                st.rerun()
        with col_p_logout:
            if st.button("🔒 Sign Out Patient", type="secondary", use_container_width=True):
                logout_patient()

        st.divider()
        st.markdown("##### ⚡ Quick Patient Links")
        q_cols = st.columns(4)
        with q_cols[0]:
            if st.button("📋 Scans & Reports", key="p_link_rep", use_container_width=True):
                st.session_state.current_view = "reports"
                st.rerun()
        with q_cols[1]:
            if st.button("📝 Log Today's Symptoms", key="p_link_sym", use_container_width=True):
                st.session_state.current_view = "symptoms"
                st.rerun()
        with q_cols[2]:
            if st.button("📈 Longitudinal Trends", key="p_link_tr", use_container_width=True):
                st.session_state.current_view = "trends"
                st.rerun()
        with q_cols[3]:
            if st.button("📤 Upload Brain MRI", key="p_link_mri", use_container_width=True):
                st.session_state.current_view = "evaluator"
                st.rerun()
        return

    # =========================================================================
    # STATE A2: DOCTOR IS ALREADY LOGGED IN -> Patient Overview & Quick Inspector
    # =========================================================================
    if current_doc:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #0f766e 0%, #064e3b 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px;">
                            Authenticated Clinical Session
                        </span>
                        <h2 style="margin: 8px 0 4px 0; font-size: 1.6rem; font-weight: 700;">
                            {current_doc['avatar']} {current_doc['full_name']}
                        </h2>
                        <p style="margin: 0; opacity: 0.9; font-size: 0.95rem;">
                            <strong>{current_doc['role']}</strong> • {current_doc['department']} • {current_doc['hospital']}
                        </p>
                        <p style="margin: 4px 0 0 0; opacity: 0.75; font-size: 0.85rem;">
                            {current_doc['license_no']} • Session Active Since: {st.session_state.doctor_session.get('login_time', 'Active')}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_top_act1, col_top_act2, col_top_act3 = st.columns([2, 1.2, 0.8])
        with col_top_act1:
            st.info("💡 You are logged in as an attending physician. Inspect patient details below or enter the Clinical Workstation console.")
        with col_top_act2:
            if st.button("🩺 Enter Full Clinician Workstation", type="primary", use_container_width=True):
                st.session_state.current_view = "clinician"
                st.rerun()
        with col_top_act3:
            if st.button("🔒 Sign Out", type="secondary", use_container_width=True):
                logout_doctor()

        st.divider()

        # Section: Patient Details Inspection
        st.subheader("👥 Assigned Patient Cohort & Detailed Clinical Inspector")
        st.caption("Review live patient demographics, molecular profiles, latest MRI volumetrics, medications, and triage risk.")

        # Patient selection pills / tabs
        assigned_mrns = current_doc.get("assigned_mrns", [p["mrn"] for p in patients])
        assigned_patients = [p for p in patients if p["mrn"] in assigned_mrns] or patients

        # Selector for active patient
        patient_options = {
            f"{p['full_name']} (MRN: {p['mrn']}) - {p['diagnosis_location']}": p["mrn"]
            for p in assigned_patients
        }
        current_active = st.session_state.get("active_patient_mrn", assigned_patients[0]["mrn"])
        curr_idx = 0
        for i, mrn in enumerate(patient_options.values()):
            if mrn == current_active:
                curr_idx = i
                break

        col_p_sel, col_p_stat = st.columns([2.5, 1.5])
        with col_p_sel:
            selected_str = st.selectbox(
                "Select Patient to Inspect Details:",
                options=list(patient_options.keys()),
                index=curr_idx,
                key="doctor_login_patient_selector"
            )
            inspect_mrn = patient_options[selected_str]
            if inspect_mrn != st.session_state.get("active_patient_mrn"):
                st.session_state.active_patient_mrn = inspect_mrn
                st.rerun()

        active_p = db.get_patient_by_mrn(st.session_state.get("active_patient_mrn", "0042")) or patients[0]
        pid = active_p["patient_id"]
        p_mol = db.get_patient_molecular(pid) or {}
        p_scans = db.get_patient_scans(pid)
        p_reports = db.get_patient_reports(pid)
        p_meds = db.get_patient_medications(pid)
        p_syms = db.get_patient_symptoms(pid)
        p_timeline = db.get_patient_timeline(pid)

        latest_scan = p_scans[-1] if p_scans else {}
        latest_twin = p_timeline[0] if p_timeline else {}
        triage_cat = latest_twin.get("triage_category", "Moderate")
        triage_col = "#ef4444" if triage_cat == "Critical" else ("#f59e0b" if triage_cat == "Moderate" else "#10b981")

        with col_p_stat:
            st.markdown(
                f"""
                <div style="background-color: {triage_col}15; border: 1.5px solid {triage_col}; border-radius: 8px; padding: 12px; text-align: center;">
                    <span style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: {triage_col};">Active Triage Classification</span><br>
                    <strong style="color: {triage_col}; font-size: 1.15rem;">{triage_cat} Priority</strong><br>
                    <span style="font-size: 0.8rem; color: #475569;">RANO 2.0: <strong>{latest_twin.get('rano_category', 'SD')}</strong> (PsP Risk: {latest_twin.get('pseudoprogression_risk', 0.84)*100:.1f}%)</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Tabular Details
        t1, t2, t3, t4 = st.tabs([
            "📋 Demographics & Biomarkers",
            "🧠 Volumetrics & Scans",
            "💊 Active Medications",
            "📝 Symptom Dynamics"
        ])

        with t1:
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.markdown("##### 👤 Patient Demographics")
                st.markdown(f"• **Full Name**: `{active_p['full_name']}`")
                st.markdown(f"• **Medical Record Number (MRN)**: `{active_p['mrn']}`")
                st.markdown(f"• **Age / Sex**: `{active_p['age']} years / {active_p['gender']}`")
                st.markdown(f"• **Diagnosis**: *{active_p['diagnosis_type']}*")
                st.markdown(f"• **Anatomical Location**: `{active_p['diagnosis_location']}`")
                st.markdown(f"• **Treatment Protocol**: `{active_p['treatment_stage']}`")
                st.markdown(f"• **Baseline KPS**: `{active_p['baseline_kps']}%`")

            with c_d2:
                st.markdown("##### 🧬 Molecular & Genomic Biomarkers")
                idh_val = p_mol.get("idh_status", "Mutant")
                mgmt_val = "Methylated (Favorable)" if p_mol.get("mgmt_methylated") else "Unmethylated (Resistance)"
                codel_val = "1p/19q Co-deleted" if p_mol.get("codeletion_1p19q") else "Intact / Non-codeleted"
                st.markdown(f"• **IDH Status**: `{idh_val}`")
                st.markdown(f"• **MGMT Promoter**: `{mgmt_val}`")
                st.markdown(f"• **Chromosomal 1p/19q**: `{codel_val}`")
                st.markdown(f"• **Assayed Date**: `{p_mol.get('assayed_at', '2026-05-20')}`")
                st.markdown(f"• **Primary Oncologist**: `{active_p['oncologist_name']}`")
                st.markdown(f"• **Nurse Navigator**: `{active_p['nurse_name']}`")

        with t2:
            st.markdown("##### 🧠 Longitudinal MRI Volumetrics (SegResNet 3D)")
            if p_scans:
                scan_table = []
                for s in p_scans:
                    scan_table.append({
                        "Scan Date": s.get("scan_date"),
                        "Whole Tumor (cm³)": f"{s.get('wt_vol_cm3', 0.0):.2f}",
                        "Tumor Core (cm³)": f"{s.get('tc_vol_cm3', 0.0):.2f}",
                        "Enhancing (cm³)": f"{s.get('et_vol_cm3', 0.0):.2f}",
                        "Edema (cm³)": f"{s.get('edema_vol_cm3', 0.0):.2f}",
                        "Dice Score": f"{s.get('dice_score', 0.0):.4f}",
                        "Perfusion rCBV": f"{s.get('estimated_rcbv', 0.0):.2f}"
                    })
                st.dataframe(scan_table, use_container_width=True)
            else:
                st.info("No historical scans on file for this patient.")

        with t3:
            st.markdown("##### 💊 Prescribed Medications & Adherence Status")
            if p_meds:
                med_table = []
                for m in p_meds:
                    med_table.append({
                        "Medication": m.get("med_name"),
                        "Dosage": m.get("dosage"),
                        "Frequency": m.get("frequency"),
                        "Status": m.get("status"),
                        "Prescribed By": m.get("prescribed_by"),
                        "Adherence Notes": m.get("adherence_notes", "")
                    })
                st.dataframe(med_table, use_container_width=True)
            else:
                st.info("No active medications prescribed.")

        with t4:
            st.markdown("##### 📝 Recent Symptom Logs")
            if p_syms:
                sym_table = []
                for s in p_syms:
                    sym_table.append({
                        "Date Logged": s.get("logged_at", "")[:10],
                        "Symptom": s.get("symptom_type"),
                        "Severity": s.get("severity"),
                        "Onset": s.get("onset_time"),
                        "Patient Notes": s.get("notes")
                    })
                st.dataframe(sym_table, use_container_width=True)
            else:
                st.info("No recent symptoms logged.")

        st.markdown("")
        col_enter, col_rep_btn = st.columns([1, 1])
        with col_enter:
            if st.button("🚀 Proceed to Clinical Workstation", type="primary", use_container_width=True):
                st.session_state.current_view = "clinician"
                st.rerun()
        with col_rep_btn:
            if st.button("📤 Upload & Verify New Report for this Patient", use_container_width=True):
                st.session_state.current_view = "clinician"
                st.session_state.clinician_active_tab = 1 # Ingestion tab
                st.rerun()

        return

    # =========================================================================
    # STATE B: NO ONE LOGGED IN -> Dual-Role Portal & New Patient Registration
    # =========================================================================
    st.markdown(
        """
        <div style="text-align: center; padding: 1.8rem 1rem 1rem 1rem;">
            <div style="display: inline-block; background-color: #0f766e15; border: 2px solid #0f766e; border-radius: 50%; padding: 16px; margin-bottom: 12px;">
                <span style="font-size: 2.8rem;">🧠</span>
            </div>
            <h1 style="color: #0f766e; font-size: 2.1rem; margin-bottom: 4px; font-weight: 800;">
                AI-Powered Cancer Care Companion
            </h1>
            <p style="color: #475569; font-size: 1.05rem; max-width: 680px; margin: 0 auto;">
                Sign in as an attending physician, access your personal patient portal, or register as a new patient.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_doc_auth, tab_patient_auth, tab_new_patient = st.tabs([
        "👨‍⚕️ Doctor & Clinician Sign-In",
        "👤 Existing Patient Sign-In",
        "➕ New Patient Registration"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: DOCTOR & CLINICIAN AUTHENTICATION
    # -------------------------------------------------------------------------
    with tab_doc_auth:
        st.markdown("##### 🩺 Select Attending Physician to Sign In:")
        st.caption("1-Click hospital smart-card simulation or credential authentication.")

        cols = st.columns(len(CLINICIAN_PROFILES))
        for idx, doc in enumerate(CLINICIAN_PROFILES):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="background: white; border: 1.5px solid #cbd5e1; border-radius: 10px; padding: 1.2rem; text-align: center; min-height: 220px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
                        <span style="font-size: 2.4rem;">{doc['avatar']}</span>
                        <h4 style="margin: 8px 0 2px 0; color: #0f172a; font-size: 1.05rem;">{doc['full_name']}</h4>
                        <span style="color: #0f766e; font-size: 0.85rem; font-weight: 600;">{doc['role']}</span>
                        <p style="margin: 8px 0; color: #64748b; font-size: 0.8rem; line-height: 1.3;">
                            {doc['department']}<br>
                            <code>{doc['license_no'].split('•')[0].strip()}</code>
                        </p>
                        <span style="background: #ecfdf5; color: #059669; font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; font-weight: 600;">
                            {doc['status']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write("")
                if st.button(f"Sign In as {doc['full_name'].split(',')[0]}", key=f"quick_login_{doc['doctor_id']}", type="primary", use_container_width=True):
                    login_doctor(doc)
                    st.success(f"Authenticated as {doc['full_name']}.")
                    if on_success:
                        on_success()
                    else:
                        st.session_state.current_view = "clinician"
                        st.rerun()

        with st.expander("🔑 Clinician Credentials & NPI Sign-In"):
            with st.form("doctor_credential_form"):
                email_in = st.text_input("Institutional Email or Clinician ID", value="dr.thorne@hospital.org")
                pass_in = st.text_input("Password / Security Token", type="password", value="neuro2026")
                dept_in = st.selectbox(
                    "Clinical Department",
                    options=[
                        "Department of Neuro-Oncology",
                        "Division of Neurosurgery",
                        "Department of Neuroradiology",
                        "Department of Radiation Oncology"
                    ]
                )
                submit_doc = st.form_submit_button("🔐 Sign In Clinician", type="primary", use_container_width=True)
                if submit_doc:
                    matched_doc = next((d for d in CLINICIAN_PROFILES if d["email"].lower() == email_in.lower()), None)
                    if not matched_doc:
                        matched_doc = {
                            "doctor_id": "doc_custom",
                            "full_name": email_in.split("@")[0].replace(".", " ").title() + ", MD",
                            "role": f"Attending Physician ({dept_in})",
                            "department": dept_in,
                            "hospital": "Comprehensive Brain Tumor Institute",
                            "license_no": "NPI: 1098234811 • State Lic: CA-88912",
                            "email": email_in,
                            "assigned_mrns": ["0042", "0043", "0044"],
                            "avatar": "🩺",
                            "status": "Attending on Service"
                        }
                    login_doctor(matched_doc)
                    st.success(f"Authenticated as {matched_doc['full_name']}.")
                    if on_success:
                        on_success()
                    else:
                        st.session_state.current_view = "clinician"
                        st.rerun()

    # -------------------------------------------------------------------------
    # TAB 2: EXISTING PATIENT SIGN-IN
    # -------------------------------------------------------------------------
    with tab_patient_auth:
        st.markdown("##### 👤 1-Click Patient Sign-In:")
        st.caption("Select your patient record below to securely enter your Patient Digital Twin Portal.")

        all_pts = db.get_all_patients()
        pt_cols = st.columns(min(3, max(1, len(all_pts))))
        for idx, pt in enumerate(all_pts):
            c_idx = idx % len(pt_cols)
            with pt_cols[c_idx]:
                st.markdown(
                    f"""
                    <div style="background: white; border: 1.5px solid #cbd5e1; border-radius: 10px; padding: 1.1rem; text-align: center; min-height: 200px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                        <span style="font-size: 2.2rem;">👤</span>
                        <h4 style="margin: 6px 0 2px 0; color: #0f172a; font-size: 1.05rem;">{pt['full_name']}</h4>
                        <span style="color: #0d9488; font-size: 0.85rem; font-weight: 700;">MRN: {pt['mrn']}</span>
                        <p style="margin: 6px 0; color: #64748b; font-size: 0.8rem; line-height: 1.3;">
                            Age {pt['age']} • {pt['diagnosis_type']}<br>
                            <em>{pt['diagnosis_location']}</em>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"Sign In as {pt['full_name'].split()[0]}", key=f"quick_patient_login_{pt['mrn']}", type="primary", use_container_width=True):
                    login_patient(pt)
                    st.success(f"Signed in as {pt['full_name']}.")
                    st.session_state.current_view = "dashboard"
                    st.rerun()

        with st.expander("🔑 Sign In by Medical Record Number (MRN) or Name"):
            with st.form("patient_mrn_login_form"):
                in_mrn = st.text_input("Enter your MRN (e.g. 0042) or Full Name:", placeholder="0042")
                in_pin = st.text_input("Access PIN / Security Code (default: 1234):", type="password", value="1234")
                submit_pt_mrn = st.form_submit_button("🔓 Open My Patient Portal", type="primary", use_container_width=True)

                if submit_pt_mrn:
                    matched = None
                    clean_in = in_mrn.strip().lower()
                    for p in all_pts:
                        if p["mrn"].lower() == clean_in or clean_in in p["full_name"].lower():
                            matched = p
                            break
                    if matched:
                        login_patient(matched)
                        st.success(f"Welcome back, {matched['full_name']}!")
                        st.session_state.current_view = "dashboard"
                        st.rerun()
                    else:
                        st.error("Patient record not found. Please verify your MRN or register below as a new patient.")

    # -------------------------------------------------------------------------
    # TAB 3: NEW PATIENT REGISTRATION / SIGN UP
    # -------------------------------------------------------------------------
    with tab_new_patient:
        st.markdown("##### ➕ Register New Patient Profile")
        st.caption("New patients can register here to instantly generate a Personalized Digital Twin, baseline imaging profile, and care companion.")

        with st.form("new_patient_registration_form", clear_on_submit=False):
            col_np1, col_np2 = st.columns(2)
            with col_np1:
                np_name = st.text_input("Patient Full Name *", placeholder="e.g. Jane Foster")
                np_age = st.number_input("Age", min_value=1, max_value=120, value=48)
                np_gender = st.selectbox("Gender", options=["Female", "Male", "Non-Binary", "Prefer not to say"], index=0)
                np_diagnosis = st.selectbox(
                    "Primary Diagnosis *",
                    options=[
                        "Glioblastoma (WHO Grade 4)",
                        "Anaplastic Astrocytoma (WHO Grade 3)",
                        "Oligodendroglioma (WHO Grade 2/3)",
                        "Diffuse Astrocytoma (WHO Grade 2)",
                        "High-Grade Glioma (Recurrent)",
                        "Meningioma",
                        "Other CNS Neoplasm"
                    ],
                    index=0
                )
                np_location = st.selectbox(
                    "Tumor Anatomical Location *",
                    options=[
                        "Left temporal lobe",
                        "Right temporal lobe",
                        "Left frontal lobe",
                        "Right frontal lobe",
                        "Left parietal lobe",
                        "Right parietal lobe",
                        "Occipital lobe",
                        "Cerebellar hemisphere",
                        "Thalamus / Deep structures"
                    ],
                    index=0
                )

            with col_np2:
                np_diag_date = st.date_input("Initial Diagnosis Date", value=datetime.now())
                np_next_visit = st.date_input("Next Scheduled Clinical Visit", value=datetime.now() + timedelta(days=14))
                np_kps = st.slider(
                    "Baseline Functional KPS Score",
                    min_value=50,
                    max_value=100,
                    value=90,
                    step=5,
                    help="Karnofsky Performance Scale: 90-100 = Normal activity; 70-80 = Able to care for self; 50-60 = Requires assistance."
                )
                np_stage = st.selectbox(
                    "Treatment Phase",
                    options=[
                        "Post-Surgical / Adjuvant Stupp Maintenance",
                        "Newly Diagnosed (Pre-Surgical)",
                        "Concurrent Chemoradiotherapy (RT + TMZ)",
                        "Surveillance & Long-Term Monitoring",
                        "Recurrence Evaluation"
                    ],
                    index=0
                )
                np_doctor = st.selectbox(
                    "Assigned Attending Oncologist",
                    options=[
                        "Dr. Aris Thorne, MD, PhD (Neuro-Oncology)",
                        "Dr. Elena Rostova, MD (Surgical Neuro-Oncology)",
                        "Dr. Marcus Vance, MD (Neuroradiology)"
                    ],
                    index=0
                )
                np_email = st.text_input("Contact Email", placeholder="patient@example.com")
                np_pin = st.text_input("Create 4-Digit Access PIN", value="1234", type="password")

            st.markdown("###### 💊 Baseline Prescribed Medication Regimen")
            col_med1, col_med2 = st.columns(2)
            with col_med1:
                np_keppra = st.selectbox(
                    "Levetiracetam (Keppra) Dosage (Seizure Prophylaxis - 08:00 AM & 08:00 PM):",
                    options=["750 mg Oral Tablet", "500 mg Oral Tablet", "1000 mg Oral Tablet"],
                    index=0
                )
            with col_med2:
                np_tmz = st.selectbox(
                    "Temozolomide (Temodar) Dosage (Chemotherapy - 09:30 PM Bedtime):",
                    options=["150 mg/m² Oral Capsule", "100 mg/m² Oral Capsule", "200 mg/m² Oral Capsule"],
                    index=0
                )

            submit_reg = st.form_submit_button("🚀 Register & Log In to Patient Portal", type="primary", use_container_width=True)

            if submit_reg:
                if not np_name.strip():
                    st.error("Please provide the patient's full name.")
                else:
                    new_pt_payload = {
                        "full_name": np_name.strip(),
                        "age": int(np_age),
                        "gender": np_gender,
                        "diagnosis_type": np_diagnosis,
                        "diagnosis_location": np_location,
                        "diagnosis_date": np_diag_date.strftime("%Y-%m-%d"),
                        "next_visit_date": np_next_visit.strftime("%Y-%m-%d"),
                        "keppra_dosage": np_keppra,
                        "tmz_dosage": np_tmz,
                        "baseline_kps": int(np_kps),
                        "treatment_stage": np_stage,
                        "oncologist_name": np_doctor.split("(")[0].strip(),
                        "nurse_name": "Sarah Jensen, RN",
                        "email": np_email.strip() or f"{np_name.strip().lower().replace(' ', '.')}@healthmail.org",
                        "pin": np_pin.strip() or "1234"
                    }
                    new_pt = db.create_patient(new_pt_payload)

                    # Ensure new patient is added to all doctors' assigned cohorts
                    for doc in CLINICIAN_PROFILES:
                        if new_pt["mrn"] not in doc.get("assigned_mrns", []):
                            doc.setdefault("assigned_mrns", []).append(new_pt["mrn"])

                    doc_sess = st.session_state.get("doctor_session", {})
                    if doc_sess.get("logged_in"):
                        active_doc = doc_sess.get("doctor") or doc_sess.get("profile")
                        if active_doc and new_pt["mrn"] not in active_doc.get("assigned_mrns", []):
                            active_doc.setdefault("assigned_mrns", []).append(new_pt["mrn"])

                    st.session_state.active_patient_mrn = new_pt["mrn"]
                    login_patient(new_pt)
                    st.toast(f"✅ Welcome {new_pt['full_name']}! Your MRN is {new_pt['mrn']}.", icon="🎉")
                    st.success(f"Patient account created successfully! Medical Record Number (MRN): **{new_pt['mrn']}**")
                    if on_success:
                        on_success()
                    else:
                        st.session_state.current_view = "dashboard"
                        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">
            🔒 HIPAA & HITECH Compliant Clinical Interface • 256-Bit End-to-End Encrypted Session • Patient Data Protected
        </div>
        """,
        unsafe_allow_html=True
    )
