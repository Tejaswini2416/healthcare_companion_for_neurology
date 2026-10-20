"""
Database Client Module (Supabase + Local Persistent Fallback)
Provides seamless dual-mode connectivity: connects directly to live Supabase PostgreSQL
if credentials exist in environment variables, or operates an in-memory/JSON-persisted
database engine for instant, reliable local and offline demonstration.
Pre-seeds multiple realistic clinical patient profiles (MRN 0042, 0043, 0044).
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union

try:
    from supabase import create_client, Client
    SUPABASE_SDK_AVAILABLE = True
except ImportError:
    SUPABASE_SDK_AVAILABLE = False


def discover_supabase_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Intelligently discovers Supabase credentials from Streamlit session state,
    Streamlit secrets.toml, environment variables, or .env file.
    Filters out dummy placeholder values.
    """
    url: Optional[str] = None
    key: Optional[str] = None

    def _is_placeholder(val: Optional[str]) -> bool:
        if not val:
            return True
        v = str(val).strip().lower()
        return (
            len(v) == 0
            or "your-project" in v
            or "xyzcompany" in v
            or "dummy" in v
            or "your-anon" in v
            or "example.com" in v
        )

    # 1. Streamlit Session State (Runtime user config via UI)
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            if "custom_supabase_url" in st.session_state and st.session_state["custom_supabase_url"]:
                candidate = str(st.session_state["custom_supabase_url"]).strip()
                if not _is_placeholder(candidate):
                    url = candidate
            if "custom_supabase_key" in st.session_state and st.session_state["custom_supabase_key"]:
                candidate = str(st.session_state["custom_supabase_key"]).strip()
                if not _is_placeholder(candidate):
                    key = candidate
    except Exception:
        pass

    # 2. Streamlit Secrets (.streamlit/secrets.toml)
    if not url or not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if not url:
                    if "SUPABASE_URL" in st.secrets:
                        candidate = str(st.secrets["SUPABASE_URL"]).strip()
                        if not _is_placeholder(candidate):
                            url = candidate
                    elif "supabase" in st.secrets and "url" in st.secrets["supabase"]:
                        candidate = str(st.secrets["supabase"]["url"]).strip()
                        if not _is_placeholder(candidate):
                            url = candidate
                if not key:
                    if "SUPABASE_KEY" in st.secrets:
                        candidate = str(st.secrets["SUPABASE_KEY"]).strip()
                        if not _is_placeholder(candidate):
                            key = candidate
                    elif "supabase" in st.secrets and "key" in st.secrets["supabase"]:
                        candidate = str(st.secrets["supabase"]["key"]).strip()
                        if not _is_placeholder(candidate):
                            key = candidate
        except Exception:
            pass

    # 3. Process Environment Variables
    if not url:
        candidate = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        if candidate and not _is_placeholder(candidate):
            url = candidate.strip()
    if not key:
        candidate = os.getenv("SUPABASE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        if candidate and not _is_placeholder(candidate):
            key = candidate.strip()

    # 4. Fallback: Parse .env file if still missing or placeholder
    env_candidate_paths = [
        ".env",
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
    ]
    for env_path in env_candidate_paths:
        if (not url or not key) and os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k in ("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL") and not url and not _is_placeholder(v):
                            url = v
                        elif k in ("SUPABASE_KEY", "NEXT_PUBLIC_SUPABASE_ANON_KEY") and not key and not _is_placeholder(v):
                            key = v
            except Exception:
                pass

    return url, key


class DatabaseClient:
    """
    Unified database interface supporting direct live Supabase PostgreSQL
    and resilient local memory storage.
    """
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        discovered_url, discovered_key = discover_supabase_credentials()
        self.url = (url or discovered_url or "").strip()
        self.key = (key or discovered_key or "").strip()
        self.is_connected_to_supabase = False
        self.supabase_client: Optional[Any] = None
        self.connection_error: Optional[str] = None
        self.last_sync_time: Optional[str] = None

        # Check if real Supabase credentials are provided
        if SUPABASE_SDK_AVAILABLE and self.url and self.key and not self.url.startswith("https://your-project"):
            self.connect_to_supabase(self.url, self.key)

        # In-memory storage tables
        self.storage: Dict[str, List[Dict[str, Any]]] = {
            "patients": [],
            "molecular_profiles": [],
            "mri_scans": [],
            "clinical_reports": [],
            "medications": [],
            "symptom_logs": [],
            "twin_timeline": [],
            "counterfactual_simulations": [],
            "appointments": [],
            "chat_conversations": []
        }

        # Seed full cohort of patient profiles for local resilient mode
        self._seed_default_cohort()

    def connect_to_supabase(self, url: str, key: str) -> tuple[bool, str]:
        """
        Dynamically connects or switches connection to a live Supabase PostgreSQL instance.
        Verifies connectivity by executing a probe query on public.patients.
        """
        url = url.strip()
        key = key.strip()
        if not url or not key:
            self.connection_error = "Project URL and API Key must not be empty."
            self.is_connected_to_supabase = False
            return False, self.connection_error

        if url.startswith("https://your-project") or "xyzcompany" in url:
            self.connection_error = "Please replace the placeholder URL with your real Supabase project URL."
            self.is_connected_to_supabase = False
            return False, self.connection_error

        if not SUPABASE_SDK_AVAILABLE:
            self.connection_error = "supabase Python package is not installed."
            self.is_connected_to_supabase = False
            return False, self.connection_error

        try:
            client = create_client(url, key)
            # Lightweight probe query
            res = client.table("patients").select("patient_id").limit(1).execute()
            self.supabase_client = client
            self.url = url
            self.key = key
            self.is_connected_to_supabase = True
            self.connection_error = None
            self.last_sync_time = datetime.now(timezone.utc).isoformat()
            print(f"[DatabaseClient] Connected successfully to remote Supabase PostgreSQL: {url}")
            return True, "Connected successfully to live Supabase PostgreSQL database."
        except Exception as e:
            err_msg = str(e)
            # If relation does not exist or schema cache doesn't have it yet, the credentials are valid but tables haven't been created yet
            if (
                "relation \"public.patients\" does not exist" in err_msg
                or "relation 'public.patients' does not exist" in err_msg
                or "PGRST204" in err_msg
                or "PGRST205" in err_msg
                or "schema cache" in err_msg
            ):
                self.supabase_client = client
                self.url = url
                self.key = key
                self.is_connected_to_supabase = True
                self.connection_error = "Connected to Supabase project, but public schema tables need to be created. Please run database/supabase_schema.sql in your Supabase SQL Editor."
                self.last_sync_time = datetime.now(timezone.utc).isoformat()
                print(f"[DatabaseClient] Connected to remote Supabase ({url}), schema tables pending setup.")
                return True, self.connection_error

            self.connection_error = f"Connection failed: {err_msg}"
            self.is_connected_to_supabase = False
            print(f"[DatabaseClient] Remote Supabase connection failed ({err_msg}). Reverting to local resilient storage.")
            return False, self.connection_error

    def disconnect_from_supabase(self):
        """Disconnects from remote Supabase and reverts to local resilient mode."""
        self.supabase_client = None
        self.is_connected_to_supabase = False
        self.connection_error = None

    def get_connection_status(self) -> Dict[str, Any]:
        """Returns real-time connection status, database telemetry, and record counts."""
        masked_url = None
        if self.url and not self.url.startswith("https://your-project"):
            if len(self.url) > 22:
                masked_url = self.url[:12] + "..." + self.url[-8:]
            else:
                masked_url = self.url

        status: Dict[str, Any] = {
            "is_connected": self.is_connected_to_supabase,
            "mode": "Live Supabase PostgreSQL (Remote)" if self.is_connected_to_supabase else "Local Resilient Engine (Pre-seeded)",
            "url": masked_url,
            "raw_url": self.url if not self.url.startswith("https://your-project") else "",
            "sdk_available": SUPABASE_SDK_AVAILABLE,
            "last_checked": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "last_sync_time": self.last_sync_time,
            "connection_error": self.connection_error,
            "counts": {}
        }

        # Query live table counts if connected
        if self.is_connected_to_supabase and self.supabase_client:
            for tbl in ["patients", "mri_scans", "symptom_logs", "clinical_reports", "medications"]:
                try:
                    res = self.supabase_client.table(tbl).select("patient_id", count="exact").limit(1).execute()
                    status["counts"][tbl] = res.count if hasattr(res, "count") and res.count is not None else len(res.data or [])
                except Exception:
                    status["counts"][tbl] = len(self.storage.get(tbl, []))
        else:
            for tbl in ["patients", "mri_scans", "symptom_logs", "clinical_reports", "medications"]:
                status["counts"][tbl] = len(self.storage.get(tbl, []))

        return status

    def seed_remote_supabase(self) -> Dict[str, Any]:
        """
        Seeds the connected remote Supabase PostgreSQL database with the full clinical cohort
        (patients, molecular profiles, MRI scans, reports, medications, symptom logs, timeline, simulations).
        """
        if not self.is_connected_to_supabase or not self.supabase_client:
            return {"success": False, "error": "Not connected to a live Supabase instance."}

        results: Dict[str, Any] = {"success": True, "seeded_tables": {}, "errors": []}
        order_of_tables = [
            "patients",
            "molecular_profiles",
            "mri_scans",
            "clinical_reports",
            "medications",
            "symptom_logs",
            "twin_timeline",
            "counterfactual_simulations"
        ]

        for table_name in order_of_tables:
            items = self.storage.get(table_name, [])
            if not items:
                continue
            try:
                cleaned_items = []
                for item in items:
                    row = dict(item)
                    if table_name == "symptom_logs":
                        # Ensure onset_time is a valid ISO timestamptz for Postgres
                        raw_onset = row.get("onset_time")
                        try:
                            datetime.fromisoformat(str(raw_onset).replace("Z", "+00:00"))
                        except Exception:
                            row["onset_time"] = row.get("logged_at", datetime.now(timezone.utc).isoformat())
                    cleaned_items.append(row)

                _ = self.supabase_client.table(table_name).upsert(cleaned_items).execute()
                results["seeded_tables"][table_name] = len(cleaned_items)
            except Exception as e:
                results["errors"].append(f"{table_name}: {str(e)}")
                results["success"] = False

        self.last_sync_time = datetime.now(timezone.utc).isoformat()
        return results

    def _seed_default_cohort(self):
        """Seeds demo data for Patient V. Thanuja (0042), Marcus Chen (0043), and Priya Sharma (0044)"""
        p42_id = "e5b38d38-2c26-4d2b-91c6-2c1b97b00042"
        p43_id = "e5b38d38-2c26-4d2b-91c6-2c1b97b00043"
        p44_id = "e5b38d38-2c26-4d2b-91c6-2c1b97b00044"
        p45_id = "e5b38d38-2c26-4d2b-91c6-2c1b97b00045"

        # 1. Patient Demographics
        self.storage["patients"] = [
            {
                "patient_id": p42_id,
                "mrn": "0042",
                "full_name": "V. Thanuja",
                "age": 24,
                "gender": "Female",
                "diagnosis_type": "High-Grade Glioma (Astrocytoma, IDH-Mutant Grade 4)",
                "diagnosis_location": "Left Temporal Lobe",
                "diagnosis_date": "2026-05-14",
                "treatment_stage": "Adjuvant Chemoradiation (Stupp Protocol Cycle 3)",
                "baseline_kps": 90,
                "oncologist_name": "Dr. Aris Thorne, MD (Neuro-Oncology)",
                "nurse_name": "Sarah Jensen, RN (Care Navigator)",
                "share_research_consent": True,
                "created_at": "2026-05-14T08:00:00Z"
            },
            {
                "patient_id": p43_id,
                "mrn": "0043",
                "full_name": "Marcus Chen",
                "age": 58,
                "gender": "Male",
                "diagnosis_type": "Glioblastoma (IDH-Wildtype WHO Grade 4)",
                "diagnosis_location": "Right Frontal Lobe",
                "diagnosis_date": "2026-07-02",
                "treatment_stage": "Post-Surgical Adjuvant Temozolomide Cycle 1",
                "baseline_kps": 80,
                "oncologist_name": "Dr. Elena Rostova, MD (Neuro-Oncology)",
                "nurse_name": "James Miller, RN (Care Navigator)",
                "share_research_consent": True,
                "created_at": "2026-07-02T08:00:00Z"
            },
            {
                "patient_id": p44_id,
                "mrn": "0044",
                "full_name": "Priya Sharma",
                "age": 41,
                "gender": "Female",
                "diagnosis_type": "Oligodendroglioma (IDH-Mutant, 1p/19q-codeleted WHO Grade 3)",
                "diagnosis_location": "Right Parietal Lobe",
                "diagnosis_date": "2025-11-20",
                "treatment_stage": "Surveillance Post-PCV Chemotherapy",
                "baseline_kps": 95,
                "oncologist_name": "Dr. Aris Thorne, MD (Neuro-Oncology)",
                "nurse_name": "Sarah Jensen, RN (Care Navigator)",
                "share_research_consent": True,
                "created_at": "2025-11-20T08:00:00Z"
            },
            {
                "patient_id": p45_id,
                "mrn": "0045",
                "full_name": "Teju",
                "age": 21,
                "gender": "Female",
                "diagnosis_type": "Glioblastoma (Astrocytoma, IDH-Mutant Grade 4)",
                "diagnosis_location": "Left Temporal Lobe",
                "diagnosis_date": "2026-06-10",
                "treatment_stage": "Adjuvant Chemoradiation (Stupp Protocol Cycle 2)",
                "baseline_kps": 90,
                "oncologist_name": "Dr. Aris Thorne, MD (Neuro-Oncology)",
                "nurse_name": "Sarah Jensen, RN (Care Navigator)",
                "share_research_consent": True,
                "created_at": "2026-06-10T08:00:00Z"
            }
        ]

        # 2. Molecular Profiles
        self.storage["molecular_profiles"] = [
            {
                "profile_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00001",
                "patient_id": p42_id,
                "idh_status": "Mutant",
                "mgmt_methylated": True,
                "codeletion_1p19q": False,
                "assayed_at": "2026-05-20"
            },
            {
                "profile_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00002",
                "patient_id": p43_id,
                "idh_status": "Wild-Type",
                "mgmt_methylated": False,
                "codeletion_1p19q": False,
                "assayed_at": "2026-07-10"
            },
            {
                "profile_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00003",
                "patient_id": p44_id,
                "idh_status": "Mutant",
                "mgmt_methylated": True,
                "codeletion_1p19q": True,
                "assayed_at": "2025-12-01"
            },
            {
                "profile_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00004",
                "patient_id": p45_id,
                "idh_status": "Mutant",
                "mgmt_methylated": True,
                "codeletion_1p19q": False,
                "assayed_at": "2026-06-15"
            }
        ]

        # 3. Longitudinal Scans
        self.storage["mri_scans"] = [
            # Patient 0042
            {
                "scan_id": "a1b38d38-2c26-4d2b-91c6-2c1b97b00011",
                "patient_id": p42_id,
                "scan_date": "2026-06-10",
                "wt_vol_cm3": 22.40,
                "tc_vol_cm3": 14.80,
                "et_vol_cm3": 8.20,
                "edema_vol_cm3": 7.60,
                "dice_score": 0.9120,
                "estimated_rcbv": 2.15,
                "mask_storage_path": "scans/0042/20260610_seg.nii.gz"
            },
            {
                "scan_id": "a2b38d38-2c26-4d2b-91c6-2c1b97b00012",
                "patient_id": p42_id,
                "scan_date": "2026-07-28",
                "wt_vol_cm3": 17.10,
                "tc_vol_cm3": 10.40,
                "et_vol_cm3": 5.60,
                "edema_vol_cm3": 6.70,
                "dice_score": 0.9340,
                "estimated_rcbv": 1.82,
                "mask_storage_path": "scans/0042/20260728_seg.nii.gz"
            },
            {
                "scan_id": "a3b38d38-2c26-4d2b-91c6-2c1b97b00013",
                "patient_id": p42_id,
                "scan_date": "2026-09-18",
                "wt_vol_cm3": 14.60,
                "tc_vol_cm3": 8.10,
                "et_vol_cm3": 4.10,
                "edema_vol_cm3": 6.50,
                "dice_score": 0.9410,
                "estimated_rcbv": 1.48,
                "mask_storage_path": "scans/0042/20260918_seg.nii.gz"
            },
            # Patient 0043
            {
                "scan_id": "a4b38d38-2c26-4d2b-91c6-2c1b97b00014",
                "patient_id": p43_id,
                "scan_date": "2026-07-05",
                "wt_vol_cm3": 38.60,
                "tc_vol_cm3": 26.20,
                "et_vol_cm3": 15.80,
                "edema_vol_cm3": 12.40,
                "dice_score": 0.9050,
                "estimated_rcbv": 3.40,
                "mask_storage_path": "scans/0043/20260705_seg.nii.gz"
            },
            {
                "scan_id": "a5b38d38-2c26-4d2b-91c6-2c1b97b00015",
                "patient_id": p43_id,
                "scan_date": "2026-09-02",
                "wt_vol_cm3": 29.40,
                "tc_vol_cm3": 18.10,
                "et_vol_cm3": 10.90,
                "edema_vol_cm3": 11.30,
                "dice_score": 0.9280,
                "estimated_rcbv": 2.75,
                "mask_storage_path": "scans/0043/20260902_seg.nii.gz"
            },
            # Patient 0044
            {
                "scan_id": "a6b38d38-2c26-4d2b-91c6-2c1b97b00016",
                "patient_id": p44_id,
                "scan_date": "2026-03-12",
                "wt_vol_cm3": 11.20,
                "tc_vol_cm3": 5.40,
                "et_vol_cm3": 1.80,
                "edema_vol_cm3": 5.80,
                "dice_score": 0.9420,
                "estimated_rcbv": 1.25,
                "mask_storage_path": "scans/0044/20260312_seg.nii.gz"
            },
            {
                "scan_id": "a7b38d38-2c26-4d2b-91c6-2c1b97b00017",
                "patient_id": p44_id,
                "scan_date": "2026-08-25",
                "wt_vol_cm3": 7.80,
                "tc_vol_cm3": 3.10,
                "et_vol_cm3": 0.90,
                "edema_vol_cm3": 4.70,
                "dice_score": 0.9580,
                "estimated_rcbv": 1.10,
                "mask_storage_path": "scans/0044/20260825_seg.nii.gz"
            },
            # Patient 0045 (Teju)
            {
                "scan_id": "a8b38d38-2c26-4d2b-91c6-2c1b97b00018",
                "patient_id": p45_id,
                "scan_date": "2026-06-15",
                "wt_vol_cm3": 15.20,
                "tc_vol_cm3": 7.10,
                "et_vol_cm3": 4.10,
                "edema_vol_cm3": 6.80,
                "dice_score": 0.9320,
                "estimated_rcbv": 1.55,
                "mask_storage_path": "scans/0045/20260615_seg.nii.gz"
            },
            {
                "scan_id": "a9b38d38-2c26-4d2b-91c6-2c1b97b00019",
                "patient_id": p45_id,
                "scan_date": "2026-09-18",
                "wt_vol_cm3": 12.80,
                "tc_vol_cm3": 5.60,
                "et_vol_cm3": 3.20,
                "edema_vol_cm3": 5.40,
                "dice_score": 0.9480,
                "estimated_rcbv": 1.45,
                "mask_storage_path": "scans/0045/20260918_seg.nii.gz"
            }
        ]

        # 4. Clinical Reports
        self.storage["clinical_reports"] = [
            # Patient 0042
            {
                "report_id": "b1b38d38-2c26-4d2b-91c6-2c1b97b00021",
                "patient_id": p42_id,
                "scan_id": "a3b38d38-2c26-4d2b-91c6-2c1b97b00013",
                "uploaded_by": "Clinician",
                "report_date": "2026-09-19",
                "raw_text": (
                    "FINDINGS: Interval reduction in enhancing residual lesion along posterior resection margin in left temporal lobe, "
                    "measuring 1.4 x 1.8 cm. Enhancing tumor volume estimated at 4.10 cm3. Moderate FLAIR hyperintensity consistent "
                    "with perilesional vasogenic edema stable at 6.50 cm3. No significant mass effect or midline shift. Ventricles are symmetric. "
                    "Relative CBV is reduced at 1.48. IMPRESSION: Consistent with post-treatment radiation effect / pseudoprogression."
                ),
                "parsed_entities": {
                    "tumor_location": "Left temporal lobe (resection margin)",
                    "enhancing_dimensions_cm": "1.4 x 1.8",
                    "edema_status": "Stable (6.50 cm3)",
                    "mass_effect": False,
                    "midline_shift_mm": 0.0,
                    "rcbv": 1.48,
                    "impression": "Treatment-induced radiation necrosis / pseudoprogression"
                },
                "plain_language_summary": (
                    "Your latest brain scan shows encouraging progress! The active area of your treated tumor has decreased in size to 4.1 cubic centimeters. "
                    "The mild tissue swelling (edema) around the surgery area is completely stable and not pushing against any healthy brain regions. "
                    "The low blood flow measurement indicates that the changes seen are safe, expected healing reactions from your radiation therapy (pseudoprogression), "
                    "rather than active cancer growth."
                ),
                "severity_tag": "Moderate",
                "created_at": "2026-09-19T14:30:00Z"
            },
            # Patient 0043
            {
                "report_id": "b2b38d38-2c26-4d2b-91c6-2c1b97b00022",
                "patient_id": p43_id,
                "scan_id": "a5b38d38-2c26-4d2b-91c6-2c1b97b00015",
                "uploaded_by": "Clinician",
                "report_date": "2026-09-03",
                "raw_text": (
                    "FINDINGS: Post-resection cavity in right frontal lobe with residual peripheral enhancement measuring 2.1 x 2.4 cm. "
                    "Mild reduction in overall volume. Surrounding FLAIR hyperintensity stable. No midline shift. Perfusion rCBV elevated at 2.75. "
                    "IMPRESSION: Consistent with residual active tumor responding partially to early post-operative adjuvant therapy."
                ),
                "parsed_entities": {
                    "tumor_location": "Right frontal lobe",
                    "enhancing_dimensions_cm": "2.1 x 2.4",
                    "edema_status": "Stable (11.30 cm3)",
                    "mass_effect": False,
                    "midline_shift_mm": 0.0,
                    "rcbv": 2.75,
                    "impression": "Residual active high-grade tumor"
                },
                "plain_language_summary": (
                    "Your recent scan shows that the surgery was effective and your treatment is stabilizing the area. "
                    "The remaining edge of the tumor in the right front portion of your brain has gotten smaller, down to 10.9 cubic centimeters. "
                    "We will continue active monitoring and your scheduled chemotherapy cycle."
                ),
                "severity_tag": "Moderate",
                "created_at": "2026-09-03T11:00:00Z"
            },
            # Patient 0044
            {
                "report_id": "b3b38d38-2c26-4d2b-91c6-2c1b97b00023",
                "patient_id": p44_id,
                "scan_id": "a7b38d38-2c26-4d2b-91c6-2c1b97b00017",
                "uploaded_by": "Clinician",
                "report_date": "2026-08-26",
                "raw_text": (
                    "FINDINGS: Continued interval involution of right parietal non-enhancing lesion following completion of PCV protocol. "
                    "Minimal residual FLAIR hyperintensity measuring 4.7 cm3. No pathological contrast enhancement. Perfusion rCBV normalized at 1.10. "
                    "No mass effect. IMPRESSION: Favorable durable treatment response; no evidence of recurrent tumor."
                ),
                "parsed_entities": {
                    "tumor_location": "Right parietal lobe",
                    "enhancing_dimensions_cm": "None (non-enhancing)",
                    "edema_status": "Minimal residual FLAIR (4.70 cm3)",
                    "mass_effect": False,
                    "midline_shift_mm": 0.0,
                    "rcbv": 1.10,
                    "impression": "Durable complete response"
                },
                "plain_language_summary": (
                    "Outstanding scan results! Your tumor continues to shrink with no signs of active cancer. "
                    "The remaining subtle scar tissue is stable and calm, with healthy normal blood flow. "
                    "Continue your current routine activities with your next check-up in 4 months."
                ),
                "severity_tag": "Mild",
                "created_at": "2026-08-26T09:15:00Z"
            },
            # Patient 0045 (Teju)
            {
                "report_id": "b4b38d38-2c26-4d2b-91c6-2c1b97b00024",
                "patient_id": p45_id,
                "scan_id": "a9b38d38-2c26-4d2b-91c6-2c1b97b00019",
                "uploaded_by": "Clinician",
                "report_date": "2026-09-18",
                "raw_text": (
                    "FINDINGS: 3T Brain MRI demonstrates stable post-surgical resection cavity in the left temporal lobe. "
                    "Residual enhancing rim measures 3.2 cm3, completely unchanged from prior examination. Vasogenic edema remains stable at 5.4 cm3. "
                    "Perfusion dynamic susceptibility contrast MRI demonstrates a normalized rCBV of 1.45, ruling out rapid tumor recurrence and "
                    "confirming treatment-related pseudoprogression under adjuvant Temozolomide. IMPRESSION: Stable pseudoprogression; excellent response to Stupp Protocol."
                ),
                "parsed_entities": {
                    "tumor_location": "Left temporal lobe",
                    "enhancing_dimensions_cm": "Residual rim (3.20 cm3)",
                    "edema_status": "Stable (5.40 cm3)",
                    "mass_effect": False,
                    "midline_shift_mm": 0.0,
                    "rcbv": 1.45,
                    "impression": "Treatment-related pseudoprogression (stable)"
                },
                "plain_language_summary": (
                    "Great news! Your brain picture shows everything is calm and stable. "
                    "The tiny spot in your left temporal lobe has calm blood flow and is sleeping quietly. "
                    "Your medicines are doing a wonderful job protecting your brain."
                ),
                "severity_tag": "Mild",
                "created_at": "2026-09-18T11:00:00Z"
            }
        ]

        # 5. Medications
        self.storage["medications"] = [
            # Patient 0042
            {
                "med_id": "c1b38d38-2c26-4d2b-91c6-2c1b97b00031",
                "patient_id": p42_id,
                "med_name": "Temozolomide (Temodar)",
                "dosage": "150 mg/m2 Oral Capsule",
                "frequency": "Nightly (Days 1-5 of 28-day cycle)",
                "instructions": "Take on empty stomach at bedtime; take antiemetic 30 min prior",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Taken",
                "adherence_notes": "Cycle 3 Day 4 completed smoothly without emesis",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-24T22:00:00Z"
            },
            {
                "med_id": "c2b38d38-2c26-4d2b-91c6-2c1b97b00032",
                "patient_id": p42_id,
                "med_name": "Levetiracetam (Keppra)",
                "dosage": "750 mg Oral Tablet",
                "frequency": "Twice Daily (Every 12 hours)",
                "instructions": "Strict seizure prophylaxis; do not abruptly discontinue",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Missed",
                "adherence_notes": "Patient missed morning dose due to gastrointestinal upset",
                "last_updated_by": "Patient",
                "updated_at": "2026-09-24T09:45:00Z"
            },
            {
                "med_id": "c3b38d38-2c26-4d2b-91c6-2c1b97b00033",
                "patient_id": p42_id,
                "med_name": "Dexamethasone",
                "dosage": "2 mg Oral Tablet",
                "frequency": "Once Daily (Morning with food)",
                "instructions": "Tapering steroid for vasogenic edema control",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Taken",
                "adherence_notes": "Steroid taper on schedule",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-25T07:30:00Z"
            },
            # Patient 0043
            {
                "med_id": "c4b38d38-2c26-4d2b-91c6-2c1b97b00034",
                "patient_id": p43_id,
                "med_name": "Temozolomide (Temodar)",
                "dosage": "200 mg/m2 Oral Capsule",
                "frequency": "Nightly (Days 1-5 of 28-day cycle)",
                "instructions": "Cycle 1 post-resection protocol",
                "prescribed_by": "Dr. Elena Rostova, MD",
                "status": "Taken",
                "adherence_notes": "Completed cycle days on schedule",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-24T21:30:00Z"
            },
            {
                "med_id": "c5b38d38-2c26-4d2b-91c6-2c1b97b00035",
                "patient_id": p43_id,
                "med_name": "Levetiracetam (Keppra)",
                "dosage": "1000 mg Oral Tablet",
                "frequency": "Twice Daily",
                "instructions": "Seizure prevention",
                "prescribed_by": "Dr. Elena Rostova, MD",
                "status": "Taken",
                "adherence_notes": "100% adherence logged",
                "last_updated_by": "Patient",
                "updated_at": "2026-09-25T08:00:00Z"
            },
            # Patient 0044
            {
                "med_id": "c6b38d38-2c26-4d2b-91c6-2c1b97b00036",
                "patient_id": p44_id,
                "med_name": "Levetiracetam (Keppra)",
                "dosage": "500 mg Oral Tablet",
                "frequency": "Twice Daily",
                "instructions": "Maintenance dose; seizure free for >12 months",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Taken",
                "adherence_notes": "Full adherence confirmed",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-25T08:00:00Z"
            },
            # Patient 0045 (Teju)
            {
                "med_id": "c7b38d38-2c26-4d2b-91c6-2c1b97b00037",
                "patient_id": p45_id,
                "med_name": "Levetiracetam (Keppra)",
                "dosage": "500 mg Oral Tablet",
                "frequency": "Twice Daily (Every 12 hours)",
                "instructions": "Strict seizure prophylaxis; take at 08:00 and 20:00 with full glass of water.",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Taken",
                "adherence_notes": "Consistent 12-hour dosing logged",
                "last_updated_by": "Patient",
                "updated_at": "2026-09-24T20:00:00Z"
            },
            {
                "med_id": "c8b38d38-2c26-4d2b-91c6-2c1b97b00038",
                "patient_id": p45_id,
                "med_name": "Temozolomide (Temodar)",
                "dosage": "150 mg/m2 Oral Capsule",
                "frequency": "Nightly at bedtime on empty stomach",
                "instructions": "Take at bedtime with water 45 min after Ondansetron antiemetic.",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Taken",
                "adherence_notes": "Adjuvant cycle maintained without nausea",
                "last_updated_by": "Doctor",
                "updated_at": "2026-09-24T22:00:00Z"
            },
            {
                "med_id": "c9b38d38-2c26-4d2b-91c6-2c1b97b00039",
                "patient_id": p45_id,
                "med_name": "Ondansetron (Zofran)",
                "dosage": "8 mg Oral Tablet",
                "frequency": "As needed (PRN) for nausea",
                "instructions": "Take 30-45 minutes before Temozolomide.",
                "prescribed_by": "Dr. Aris Thorne, MD",
                "status": "Due",
                "adherence_notes": "Available PRN",
                "last_updated_by": "Patient",
                "updated_at": "2026-09-24T21:15:00Z"
            }
        ]

        # 6. Symptom Logs
        self.storage["symptom_logs"] = [
            # Patient 0042
            {
                "log_id": "d1b38d38-2c26-4d2b-91c6-2c1b97b00041",
                "patient_id": p42_id,
                "logged_at": "2026-09-23T16:00:00Z",
                "symptom_type": "Fatigue",
                "severity": "Moderate",
                "onset_time": "Late afternoon",
                "notes": "Needed a 2 hour nap after taking morning doses."
            },
            {
                "log_id": "d2b38d38-2c26-4d2b-91c6-2c1b97b00042",
                "patient_id": p42_id,
                "logged_at": "2026-09-24T09:30:00Z",
                "symptom_type": "Seizure",
                "severity": "Moderate",
                "onset_time": "Morning 9:30 AM",
                "notes": "Felt strange olfactory sensations (burning smell) and mild right hand tingling for 45 seconds after missing morning Keppra."
            },
            {
                "log_id": "d3b38d38-2c26-4d2b-91c6-2c1b97b00043",
                "patient_id": p42_id,
                "logged_at": "2026-09-25T07:15:00Z",
                "symptom_type": "Headache",
                "severity": "Mild",
                "onset_time": "Morning 7:00 AM",
                "notes": "Dull throbbing in left temple, relieved by drinking water."
            },
            # Patient 0043
            {
                "log_id": "d4b38d38-2c26-4d2b-91c6-2c1b97b00044",
                "patient_id": p43_id,
                "logged_at": "2026-09-23T14:30:00Z",
                "symptom_type": "Fatigue",
                "severity": "Moderate",
                "onset_time": "Mid-afternoon",
                "notes": "Mild drowsiness after physical therapy."
            },
            {
                "log_id": "d5b38d38-2c26-4d2b-91c6-2c1b97b00045",
                "patient_id": p43_id,
                "logged_at": "2026-09-24T18:00:00Z",
                "symptom_type": "Speech",
                "severity": "Mild",
                "onset_time": "Evening",
                "notes": "Brief word-finding delay; fully recovered."
            },
            # Patient 0044
            {
                "log_id": "d6b38d38-2c26-4d2b-91c6-2c1b97b00046",
                "patient_id": p44_id,
                "logged_at": "2026-09-22T19:00:00Z",
                "symptom_type": "Fatigue",
                "severity": "Mild",
                "onset_time": "Evening",
                "notes": "Usual fatigue after full workday; resolved with rest."
            },
            # Patient 0045 (Teju)
            {
                "log_id": "d7b38d38-2c26-4d2b-91c6-2c1b97b00047",
                "patient_id": p45_id,
                "logged_at": "2026-09-20T14:30:00Z",
                "symptom_type": "Cognitive Fatigue",
                "severity": "Mild",
                "onset_time": "Afternoon 2:30 PM",
                "notes": "Mild fatigue after radiation session; rested for 45 minutes and felt better."
            },
            {
                "log_id": "d8b38d38-2c26-4d2b-91c6-2c1b97b00048",
                "patient_id": p45_id,
                "logged_at": "2026-09-22T09:15:00Z",
                "symptom_type": "Sensory Aura",
                "severity": "Mild",
                "onset_time": "Morning 9:15 AM",
                "notes": "Fleeting deja vu lasting ~10 seconds. No seizure activity; Keppra taken on time."
            }
        ]

        # 7. Twin Timeline
        self.storage["twin_timeline"] = [
            # Patient 0042
            {
                "timeline_id": "e1b38d38-2c26-4d2b-91c6-2c1b97b00051",
                "patient_id": p42_id,
                "evaluation_date": "2026-09-20",
                "progression_risk_score": 0.2850,
                "triage_category": "Moderate",
                "rano_category": "PsP",
                "pseudoprogression_risk": 0.8420,
                "is_pseudoprogression": True,
                "adherence_rate_pct": 85.70,
                "latent_twin_vector": [round(float(x), 4) for x in np.random.uniform(-0.15, 0.15, 64)],
                "alert_banner_text": "Attention: Missed Levetiracetam (Keppra) dose coincides with a newly reported sensory seizure aura. Please take your prescribed dose and contact your care team.",
                "created_at": "2026-09-20T10:00:00Z"
            },
            # Patient 0043
            {
                "timeline_id": "e2b38d38-2c26-4d2b-91c6-2c1b97b00052",
                "patient_id": p43_id,
                "evaluation_date": "2026-09-03",
                "progression_risk_score": 0.4120,
                "triage_category": "Moderate",
                "rano_category": "SD",
                "pseudoprogression_risk": 0.2800,
                "is_pseudoprogression": False,
                "adherence_rate_pct": 100.0,
                "latent_twin_vector": [round(float(x), 4) for x in np.random.uniform(-0.15, 0.15, 64)],
                "alert_banner_text": "Treatment Adherence High: Stable post-surgical cavity volume. Monitor for any muscle weakness or speech difficulty.",
                "created_at": "2026-09-03T12:00:00Z"
            },
            # Patient 0044
            {
                "timeline_id": "e3b38d38-2c26-4d2b-91c6-2c1b97b00053",
                "patient_id": p44_id,
                "evaluation_date": "2026-08-26",
                "progression_risk_score": 0.1150,
                "triage_category": "Mild",
                "rano_category": "CR",
                "pseudoprogression_risk": 0.0500,
                "is_pseudoprogression": False,
                "adherence_rate_pct": 100.0,
                "latent_twin_vector": [round(float(x), 4) for x in np.random.uniform(-0.15, 0.15, 64)],
                "alert_banner_text": "Excellent Stability: Complete treatment response maintained with 100% medication adherence.",
                "created_at": "2026-08-26T10:00:00Z"
            },
            # Patient 0045 (Teju)
            {
                "timeline_id": "e4b38d38-2c26-4d2b-91c6-2c1b97b00054",
                "patient_id": p45_id,
                "evaluation_date": "2026-09-18",
                "progression_risk_score": 0.1820,
                "triage_category": "Moderate",
                "rano_category": "PsP (Pseudoprogression / Treatment Effect)",
                "pseudoprogression_risk": 0.8250,
                "is_pseudoprogression": True,
                "adherence_rate_pct": 100.0,
                "latent_twin_vector": [round(float(x), 4) for x in np.random.uniform(-0.15, 0.15, 64)],
                "alert_banner_text": "Stable Pseudoprogression: Tumor margins demonstrate calm response to Temozolomide with low rCBV (1.45).",
                "created_at": "2026-09-18T10:00:00Z"
            }
        ]

        # 8. Counterfactual In-Silico Simulations
        self.storage["counterfactual_simulations"] = [
            {
                "sim_id": "f1b38d38-2c26-4d2b-91c6-2c1b97b00061",
                "patient_id": p42_id,
                "simulation_name": "Standard of Care TMZ Adjuvant Continuation",
                "intervention_type": "SOC_TMZ",
                "forecast_horizon_days": 180,
                "predicted_trajectory": {
                    "days": [0, 30, 60, 90, 180],
                    "tumor_vol_cm3": [14.6, 13.9, 13.2, 12.8, 12.4],
                    "kps_score": [90, 90, 85, 85, 85],
                    "confidence_interval": [0.92, 0.88, 0.84, 0.79, 0.73]
                },
                "created_at": "2026-09-20T12:00:00Z"
            },
            {
                "sim_id": "f2b38d38-2c26-4d2b-91c6-2c1b97b00062",
                "patient_id": p42_id,
                "simulation_name": "Dose-Dense Temozolomide (7/14 Schedule)",
                "intervention_type": "DOSE_DENSE",
                "forecast_horizon_days": 180,
                "predicted_trajectory": {
                    "days": [0, 30, 60, 90, 180],
                    "tumor_vol_cm3": [14.6, 13.1, 11.8, 11.2, 10.9],
                    "kps_score": [90, 85, 80, 80, 75],
                    "confidence_interval": [0.89, 0.84, 0.80, 0.75, 0.68]
                },
                "created_at": "2026-09-20T12:05:00Z"
            },
            {
                "sim_id": "f3b38d38-2c26-4d2b-91c6-2c1b97b00063",
                "patient_id": p42_id,
                "simulation_name": "Therapeutic Hold / TMZ Pause",
                "intervention_type": "HOLD_TMZ",
                "forecast_horizon_days": 180,
                "predicted_trajectory": {
                    "days": [0, 30, 60, 90, 180],
                    "tumor_vol_cm3": [14.6, 16.4, 19.8, 24.5, 36.2],
                    "kps_score": [90, 85, 80, 70, 60],
                    "confidence_interval": [0.91, 0.86, 0.81, 0.74, 0.65]
                },
                "created_at": "2026-09-20T12:10:00Z"
            }
        ]

        # 9. Pre-Seeded Appointments
        self.storage["appointments"] = [
            # Patient 0042 (V. Thanuja)
            {
                "appt_id": "apt-0042-01",
                "patient_id": p42_id,
                "doctor_name": "Dr. Aris Thorne, MD",
                "department": "Department of Neuro-Oncology",
                "appt_date": "2026-10-28",
                "appt_time": "10:30 AM",
                "service_type": "Neuro-Oncology Follow-up & 3D Volumetric Review",
                "location": "Brain Tumor Center, Suite 4B",
                "status": "Confirmed",
                "preparation_notes": "Bring recent medication log and list of any aura/headache episodes. Continue Keppra as scheduled.",
                "created_at": "2026-09-20T10:00:00Z"
            },
            {
                "appt_id": "apt-0042-02",
                "patient_id": p42_id,
                "doctor_name": "Dr. Marcus Vance, MD",
                "department": "Department of Neuroradiology",
                "appt_date": "2026-11-15",
                "appt_time": "02:00 PM",
                "service_type": "Surveillance 3T Brain MRI with Perfusion (rCBV)",
                "location": "Advanced Neuro-Imaging Pavilion, Bay 2",
                "status": "Scheduled",
                "preparation_notes": "Fast 4 hours prior if IV gadolinium contrast is administered. Remove all metallic items.",
                "created_at": "2026-09-20T10:00:00Z"
            },
            {
                "appt_id": "apt-0042-03",
                "patient_id": p42_id,
                "doctor_name": "Sarah Jensen, RN",
                "department": "Oncology Care Navigation",
                "appt_date": "2026-10-24",
                "appt_time": "11:00 AM",
                "service_type": "Adherence & Wellness Navigation Check-in",
                "location": "Telehealth Video Consultation",
                "status": "Confirmed",
                "preparation_notes": "Log today's symptoms in the app prior to the call.",
                "created_at": "2026-09-20T10:00:00Z"
            },
            # Patient 0043 (Marcus Chen)
            {
                "appt_id": "apt-0043-01",
                "patient_id": p43_id,
                "doctor_name": "Dr. Elena Rostova, MD",
                "department": "Division of Surgical Neuro-Oncology",
                "appt_date": "2026-11-04",
                "appt_time": "09:00 AM",
                "service_type": "Surgical Margin Review & Adjuvant Cycle Assessment",
                "location": "Brain Tumor Center, Suite 2A",
                "status": "Confirmed",
                "preparation_notes": "Review CBC with differential lab results taken 48 hours prior.",
                "created_at": "2026-09-20T10:00:00Z"
            },
            # Patient 0044 (Priya Sharma)
            {
                "appt_id": "apt-0044-01",
                "patient_id": p44_id,
                "doctor_name": "Dr. Aris Thorne, MD",
                "department": "Department of Neuro-Oncology",
                "appt_date": "2026-12-02",
                "appt_time": "11:15 AM",
                "service_type": "Long-Term Surveillance Consultation Post-PCV",
                "location": "Brain Tumor Center, Suite 4B",
                "status": "Scheduled",
                "preparation_notes": "Review surveillance MRI scan with attending oncologist.",
                "created_at": "2026-09-20T10:00:00Z"
            },
            # Patient 0045 (Teju)
            {
                "appt_id": "apt-0045-01",
                "patient_id": p45_id,
                "doctor_name": "Dr. Aris Thorne, MD",
                "department": "Department of Neuro-Oncology",
                "appt_date": "2026-10-30",
                "appt_time": "10:00 AM",
                "service_type": "Cycle 2 Evaluation & KPS Assessment",
                "location": "Brain Tumor Center, Suite 4B",
                "status": "Confirmed",
                "preparation_notes": "Review symptom diary and blood counts.",
                "created_at": "2026-09-20T10:00:00Z"
            }
        ]
        self.storage["chat_conversations"] = []

    # --- Query & Retrieval APIs ---

    def _sanitize_patient_record(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all standard keys exist on patient record to prevent KeyError."""
        if not p:
            return p
        p.setdefault("diagnosis_date", p.get("created_at", "2026-05-14")[:10] if p.get("created_at") else "2026-05-14")
        p.setdefault("next_visit_date", "2026-10-28")
        p.setdefault("baseline_kps", 90)
        p.setdefault("treatment_stage", "Post-Surgical / Adjuvant Stupp Maintenance")
        p.setdefault("diagnosis_location", "Left Temporal Lobe")
        p.setdefault("diagnosis_type", "High-Grade Glioma")
        p.setdefault("oncologist_name", "Dr. Aris Thorne, MD (Neuro-Oncology)")
        p.setdefault("nurse_name", "Sarah Jensen, RN (Care Navigator)")
        return p

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Retrieves all registered patients."""
        pts = self.storage["patients"]
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("patients").select("*").order("mrn").execute()
                if res.data:
                    pts = res.data
            except Exception:
                pass
        return [self._sanitize_patient_record(dict(p)) for p in pts]

    def get_patient_by_mrn(self, mrn: str = "0042") -> Optional[Dict[str, Any]]:
        """Retrieves patient demographics by MRN."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("patients").select("*").eq("mrn", mrn).execute()
                if res.data:
                    return self._sanitize_patient_record(dict(res.data[0]))
            except Exception:
                pass
        for p in self.storage["patients"]:
            if str(p.get("mrn")) == str(mrn):
                return self._sanitize_patient_record(p)
        return None

    def get_patient_by_id(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves patient demographics by UUID."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("patients").select("*").eq("patient_id", patient_id).execute()
                if res.data:
                    return self._sanitize_patient_record(dict(res.data[0]))
            except Exception:
                pass
        for p in self.storage["patients"]:
            if p["patient_id"] == patient_id:
                return self._sanitize_patient_record(p)
        return None

    def get_patient_molecular(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves molecular profile for a patient."""
        if self.is_connected_to_supabase:
            try:
                res = self.supabase_client.table("molecular_profiles").select("*").eq("patient_id", patient_id).execute()
                if res.data:
                    return res.data[0]
            except Exception:
                pass
        for m in self.storage["molecular_profiles"]:
            if m["patient_id"] == patient_id:
                return m
        return None

    def get_patient_scans(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves historical MRI scans ordered by scan_date ascending.
        Fetches in real-time from Supabase PostgreSQL if connected, with resilient local fallback.
        """
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                res = self.supabase_client.table("mri_scans").select("*").eq("patient_id", patient_id).order("scan_date").execute()
                if res.data is not None and len(res.data) > 0:
                    cleaned_scans = []
                    for s in res.data:
                        sc = dict(s)
                        for k in ["wt_vol_cm3", "tc_vol_cm3", "et_vol_cm3", "edema_vol_cm3", "dice_score", "estimated_rcbv"]:
                            if k in sc and sc[k] is not None:
                                try:
                                    sc[k] = float(sc[k])
                                except (ValueError, TypeError):
                                    pass
                        cleaned_scans.append(sc)
                    return sorted(cleaned_scans, key=lambda x: str(x.get("scan_date", "")))
            except Exception as e:
                print(f"[DatabaseClient] Remote Supabase scan fetch warning ({e}). Reverting to local store.")

        scans = [s for s in self.storage["mri_scans"] if s["patient_id"] == patient_id]
        return sorted(scans, key=lambda x: str(x.get("scan_date", "")))

    def get_patient_reports(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves clinical reports ordered by report_date descending."""
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                res = self.supabase_client.table("clinical_reports").select("*").eq("patient_id", patient_id).order("report_date", desc=True).execute()
                if res.data is not None and len(res.data) > 0:
                    return res.data
            except Exception:
                pass
        reports = [r for r in self.storage["clinical_reports"] if r["patient_id"] == patient_id]
        return sorted(reports, key=lambda x: str(x.get("report_date", "")), reverse=True)

    def get_patient_medications(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves active medications and adherence status."""
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                res = self.supabase_client.table("medications").select("*").eq("patient_id", patient_id).execute()
                if res.data is not None and len(res.data) > 0:
                    return res.data
            except Exception:
                pass
        return [m for m in self.storage["medications"] if m["patient_id"] == patient_id]

    def get_patient_symptoms(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves patient-logged symptoms ordered by logged_at descending.
        Directly queries Supabase PostgreSQL in real-time if connected.
        """
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                res = self.supabase_client.table("symptom_logs")\
                    .select("*")\
                    .eq("patient_id", patient_id)\
                    .order("logged_at", desc=True)\
                    .limit(limit)\
                    .execute()
                if res.data is not None and len(res.data) > 0:
                    return res.data
            except Exception as e:
                print(f"[DatabaseClient] Remote Supabase symptom fetch warning ({e}). Reverting to local store.")

        syms = [s for s in self.storage["symptom_logs"] if s["patient_id"] == patient_id]
        return sorted(syms, key=lambda x: str(x.get("logged_at", "")), reverse=True)[:limit]

    def get_patient_timeline(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves latest digital twin evaluations."""
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                res = self.supabase_client.table("twin_timeline").select("*").eq("patient_id", patient_id).order("evaluation_date", desc=True).execute()
                if res.data is not None and len(res.data) > 0:
                    return res.data
            except Exception:
                pass
        t = [item for item in self.storage["twin_timeline"] if item["patient_id"] == patient_id]
        return sorted(t, key=lambda x: str(x.get("evaluation_date", "")), reverse=True)

    def get_patient_simulations(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves in-silico simulations for the patient."""
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                res = self.supabase_client.table("counterfactual_simulations").select("*").eq("patient_id", patient_id).execute()
                if res.data is not None and len(res.data) > 0:
                    return res.data
            except Exception:
                pass
        return [sim for sim in self.storage["counterfactual_simulations"] if sim["patient_id"] == patient_id]

    def _resolve_patient_id(self, patient_id_or_mrn: str) -> str:
        """Resolves patient ID whether given as a UUID, pat_0042, or MRN 0042."""
        if not patient_id_or_mrn:
            return "e5b38d38-2c26-4d2b-91c6-2c1b97b00042"
        # Direct match in patients table
        p = self.get_patient_by_id(patient_id_or_mrn) or self.get_patient_by_mrn(patient_id_or_mrn)
        if p:
            return p["patient_id"]
        # Check if contains mrn digits (e.g. pat_0042 -> 0042)
        digits = "".join([c for c in patient_id_or_mrn if c.isdigit()])
        if digits:
            p = self.get_patient_by_mrn(digits)
            if p:
                return p["patient_id"]
        return patient_id_or_mrn

    def _normalize_appointment(self, appt: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures appointment has standardized aliases: date/appt_date, clinician/doctor_name, title/service_type, appt_id/appointment_id."""
        item = dict(appt)
        if "appt_id" in item and "appointment_id" not in item:
            item["appointment_id"] = item["appt_id"]
        elif "appointment_id" in item and "appt_id" not in item:
            item["appt_id"] = item["appointment_id"]

        if "appt_date" in item and "date" not in item:
            item["date"] = item["appt_date"]
        elif "date" in item and "appt_date" not in item:
            item["appt_date"] = item["date"]

        if "doctor_name" in item and "clinician" not in item:
            item["clinician"] = item["doctor_name"]
        elif "clinician" in item and "doctor_name" not in item:
            item["doctor_name"] = item["clinician"]

        if "service_type" in item and "title" not in item:
            item["title"] = item["service_type"]
        elif "title" in item and "service_type" not in item:
            item["service_type"] = item["title"]

        if "appt_time" in item and "time" not in item:
            item["time"] = item["appt_time"]
        elif "time" in item and "appt_time" not in item:
            item["appt_time"] = item["time"]
        return item

    def get_patient_appointments(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves upcoming and scheduled appointments for a patient."""
        target_pid = self._resolve_patient_id(patient_id)
        raw_list = []
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                res = self.supabase_client.table("appointments").select("*").eq("patient_id", target_pid).order("appt_date").execute()
                if res.data is not None and len(res.data) > 0:
                    raw_list = res.data
            except Exception:
                pass
        if not raw_list:
            raw_list = [a for a in self.storage.get("appointments", []) if a.get("patient_id") in (target_pid, patient_id)]

        normalized = [self._normalize_appointment(a) for a in raw_list]
        return sorted(normalized, key=lambda x: (str(x.get("appt_date", "")), str(x.get("appt_time", ""))))

    def get_chat_history(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves stored conversation history for a patient."""
        target_pid = self._resolve_patient_id(patient_id)
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                res = self.supabase_client.table("chat_conversations").select("*").eq("patient_id", target_pid).order("created_at").limit(limit).execute()
                if res.data is not None and len(res.data) > 0:
                    return res.data
            except Exception:
                pass
        msgs = [m for m in self.storage.get("chat_conversations", []) if m.get("patient_id") in (target_pid, patient_id)]
        return sorted(msgs, key=lambda x: str(x.get("created_at", "")))[-limit:]

    # --- Mutations & Insertions ---

    def book_appointment(self, appt_data: Any, maybe_appt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Books/requests a new clinical appointment. Resilient to book_appointment(dict) or book_appointment(patient_id, dict)."""
        if maybe_appt is not None and isinstance(appt_data, str):
            actual_data = dict(maybe_appt)
            actual_data["patient_id"] = appt_data
            appt_dict = actual_data
        elif isinstance(appt_data, dict):
            appt_dict = dict(appt_data)
        else:
            appt_dict = {}

        target_pid = self._resolve_patient_id(appt_dict.get("patient_id", ""))
        appt_dict["patient_id"] = target_pid

        if "appt_id" not in appt_dict:
            appt_dict["appt_id"] = str(uuid.uuid4())
        if "created_at" not in appt_dict:
            appt_dict["created_at"] = datetime.now(timezone.utc).isoformat()
        if "status" not in appt_dict:
            appt_dict["status"] = "Confirmed"

        normalized = self._normalize_appointment(appt_dict)

        if self.is_connected_to_supabase and self.supabase_client:
            try:
                # Keep DB column names compliant
                db_record = {
                    "appt_id": normalized.get("appt_id"),
                    "patient_id": normalized.get("patient_id"),
                    "doctor_name": normalized.get("doctor_name", normalized.get("clinician")),
                    "department": normalized.get("department", "Neuro-Oncology"),
                    "appt_date": normalized.get("appt_date", normalized.get("date")),
                    "appt_time": normalized.get("appt_time", normalized.get("time", "10:00 AM")),
                    "service_type": normalized.get("service_type", normalized.get("title", "Clinical Consultation")),
                    "location": normalized.get("location", "Main Clinic"),
                    "status": normalized.get("status", "Confirmed"),
                    "preparation_notes": normalized.get("preparation_notes", ""),
                    "created_at": normalized.get("created_at")
                }
                self.supabase_client.table("appointments").insert(db_record).execute()
            except Exception as e:
                print(f"[DatabaseClient] Supabase appointment insert error: {e}")

        self.storage.setdefault("appointments", []).append(normalized)
        return normalized

    def save_chat_message(
        self,
        patient_id: str,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Saves a message to patient chat conversation history."""
        target_pid = self._resolve_patient_id(patient_id)
        msg = {
            "message_id": str(uuid.uuid4()),
            "patient_id": target_pid,
            "role": role,
            "content": content,
            "model_used": model_used or "AI Health Companion",
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("chat_conversations").insert(msg).execute()
            except Exception:
                pass
        self.storage.setdefault("chat_conversations", []).append(msg)
        return msg

    def clear_chat_history(self, patient_id: str) -> bool:
        """Clears conversation history for a patient."""
        target_pid = self._resolve_patient_id(patient_id)
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("chat_conversations").delete().eq("patient_id", target_pid).execute()
            except Exception:
                pass
        self.storage["chat_conversations"] = [
            m for m in self.storage.get("chat_conversations", []) if m.get("patient_id") not in (target_pid, patient_id)
        ]
        return True

    def log_symptom(self, symptom_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Logs a patient symptom entry with guaranteed PostgreSQL schema compliance.
        Directly inserts into public.symptom_logs in Supabase PostgreSQL when connected.
        """
        if "log_id" not in symptom_data:
            symptom_data["log_id"] = str(uuid.uuid4())
        if "logged_at" not in symptom_data:
            symptom_data["logged_at"] = datetime.now(timezone.utc).isoformat()

        # Map symptom_type to valid PostgreSQL check constraint enum: ('Headache', 'Vision', 'Seizure', 'Fatigue', 'Speech', 'Motor')
        raw_type = str(symptom_data.get("symptom_type", "Headache"))
        matched_type = "Headache"
        for valid_t in ["Headache", "Vision", "Seizure", "Fatigue", "Speech", "Motor"]:
            if valid_t.lower() in raw_type.lower():
                matched_type = valid_t
                break
        symptom_data["symptom_type"] = matched_type

        # Map severity to ('Mild', 'Moderate', 'Severe')
        raw_sev = str(symptom_data.get("severity", "Mild"))
        if "severe" in raw_sev.lower() or "4" in raw_sev or "5" in raw_sev:
            symptom_data["severity"] = "Severe"
        elif "moderate" in raw_sev.lower() or "3" in raw_sev:
            symptom_data["severity"] = "Moderate"
        else:
            symptom_data["severity"] = "Mild"

        # Ensure onset_time is a valid ISO timestamptz for Postgres
        raw_onset = symptom_data.get("onset_time")
        iso_onset = None
        if raw_onset:
            try:
                datetime.fromisoformat(str(raw_onset).replace("Z", "+00:00"))
                iso_onset = str(raw_onset)
            except Exception:
                orig_notes = symptom_data.get("notes", "")
                symptom_data["notes"] = f"[Onset: {raw_onset}] {orig_notes}".strip()
                iso_onset = datetime.now(timezone.utc).isoformat()
        else:
            iso_onset = datetime.now(timezone.utc).isoformat()
        symptom_data["onset_time"] = iso_onset

        # Insert into live Supabase PostgreSQL
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("symptom_logs").insert(symptom_data).execute()
                self.last_sync_time = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                print(f"[DatabaseClient] Supabase symptom_logs insert error: {e}")

        self.storage["symptom_logs"].insert(0, symptom_data)
        return symptom_data

    def prescribe_medication(self, med_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adds a new prescription for a patient."""
        if "med_id" not in med_data:
            med_data["med_id"] = str(uuid.uuid4())
        if "updated_at" not in med_data:
            med_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        if "status" not in med_data:
            med_data["status"] = "Due"
        if "last_updated_by" not in med_data:
            med_data["last_updated_by"] = "Doctor"

        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("medications").insert(med_data).execute()
                self.last_sync_time = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                print(f"[DatabaseClient] Supabase medications insert error: {e}")

        self.storage["medications"].append(med_data)
        return med_data

    def update_medication_status(self, med_id: str, new_status: str, notes: Optional[str] = None, updated_by: str = "Doctor") -> bool:
        """Updates medication adherence status ('Taken', 'Due', 'Missed') and clinical notes."""
        now_str = datetime.now(timezone.utc).isoformat()
        update_fields: Dict[str, Any] = {
            "status": new_status,
            "updated_at": now_str,
            "last_updated_by": updated_by
        }
        if notes:
            update_fields["adherence_notes"] = notes

        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("medications").update(update_fields).eq("med_id", med_id).execute()
                self.last_sync_time = now_str
            except Exception as e:
                print(f"[DatabaseClient] Supabase medications update error: {e}")

        for m in self.storage["medications"]:
            if m["med_id"] == med_id:
                m["status"] = new_status
                m["updated_at"] = now_str
                m["last_updated_by"] = updated_by
                if notes:
                    m["adherence_notes"] = notes
                return True
        return False

    def insert_scan_record(self, scan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inserts a segmented MRI scan record with PostgreSQL schema compliance.
        Directly inserts into public.mri_scans in Supabase PostgreSQL when connected.
        """
        if "scan_id" not in scan_data:
            scan_data["scan_id"] = str(uuid.uuid4())
        if "created_at" not in scan_data:
            scan_data["created_at"] = datetime.now(timezone.utc).isoformat()
        if "scan_date" not in scan_data:
            scan_data["scan_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Format numeric fields
        for k in ["wt_vol_cm3", "tc_vol_cm3", "et_vol_cm3", "edema_vol_cm3", "dice_score", "estimated_rcbv"]:
            if k in scan_data and scan_data[k] is not None:
                try:
                    scan_data[k] = round(float(scan_data[k]), 4 if "dice" in k else 2)
                except Exception:
                    pass

        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("mri_scans").insert(scan_data).execute()
                self.last_sync_time = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                print(f"[DatabaseClient] Supabase mri_scans insert error: {e}")

        self.storage["mri_scans"].append(scan_data)
        return scan_data

    def insert_clinical_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts a new clinical report."""
        if "report_id" not in report_data:
            report_data["report_id"] = str(uuid.uuid4())
        if "created_at" not in report_data:
            report_data["created_at"] = datetime.now(timezone.utc).isoformat()
        if "uploaded_by" not in report_data:
            report_data["uploaded_by"] = "Clinician"

        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("clinical_reports").insert(report_data).execute()
                self.last_sync_time = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                print(f"[DatabaseClient] Supabase clinical_reports insert error: {e}")

        self.storage["clinical_reports"].insert(0, report_data)
        return report_data

    def update_twin_timeline(self, timeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Appends a new digital twin evaluation milestone."""
        if "timeline_id" not in timeline_data:
            timeline_data["timeline_id"] = str(uuid.uuid4())
        if "created_at" not in timeline_data:
            timeline_data["created_at"] = datetime.now(timezone.utc).isoformat()

        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("twin_timeline").insert(timeline_data).execute()
                self.last_sync_time = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                print(f"[DatabaseClient] Supabase twin_timeline insert error: {e}")

        self.storage["twin_timeline"].insert(0, timeline_data)
        return timeline_data

    def get_connection_status(self) -> Dict[str, Any]:
        """Returns connection health and telemetry counts."""
        counts = {
            "patients": len(self.storage.get("patients", [])),
            "mri_scans": len(self.storage.get("mri_scans", [])),
            "symptom_logs": len(self.storage.get("symptom_logs", [])),
            "clinical_reports": len(self.storage.get("clinical_reports", [])),
            "medications": len(self.storage.get("medications", [])),
            "molecular_profiles": len(self.storage.get("molecular_profiles", [])),
            "twin_timeline": len(self.storage.get("twin_timeline", [])),
        }
        return {
            "is_connected": self.is_connected_to_supabase,
            "url": self.url if self.is_connected_to_supabase else "",
            "raw_url": self.url,
            "counts": counts,
            "connection_error": self.connection_error,
            "last_sync_time": self.last_sync_time
        }

    def disconnect_from_supabase(self):
        """Disconnects from remote Supabase and falls back to local storage."""
        self.is_connected_to_supabase = False
        self.supabase_client = None
        self.connection_error = None

    def seed_remote_supabase(self) -> Dict[str, Any]:
        """Seeds remote Supabase PostgreSQL with local cohort records."""
        if not self.is_connected_to_supabase or not self.supabase_client:
            return {"success": False, "seeded_tables": [], "errors": ["Not connected to Supabase."]}
        seeded = []
        errors = []
        for table_name in ["patients", "molecular_profiles", "mri_scans", "clinical_reports", "medications", "symptom_logs", "twin_timeline"]:
            records = self.storage.get(table_name, [])
            if records:
                try:
                    self.supabase_client.table(table_name).upsert(records).execute()
                    seeded.append(f"{table_name} ({len(records)} records)")
                except Exception as e:
                    errors.append(f"{table_name}: {e}")
        return {"success": len(errors) == 0, "seeded_tables": seeded, "errors": errors}

    def create_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registers a new patient, generating MRN if needed, creating molecular profile,
        baseline scan, initial prescriptions, and digital twin state.
        """
        # Determine next available MRN if not provided
        mrn = str(patient_data.get("mrn", "")).strip()
        if not mrn:
            existing_mrns = []
            for p in self.storage.get("patients", []):
                m_str = str(p.get("mrn", ""))
                if m_str.isdigit():
                    existing_mrns.append(int(m_str))
            next_mrn_int = (max(existing_mrns) + 1) if existing_mrns else 45
            mrn = f"{next_mrn_int:04d}"

        patient_id = patient_data.get("patient_id") or str(uuid.uuid4())
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        diagnosis_date = str(patient_data.get("diagnosis_date") or today_str)
        next_visit_date = str(patient_data.get("next_visit_date") or (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"))

        full_record = {
            "patient_id": patient_id,
            "mrn": mrn,
            "full_name": patient_data.get("full_name", f"Patient {mrn}"),
            "age": int(patient_data.get("age", 45)),
            "gender": patient_data.get("gender", "Female"),
            "diagnosis_type": patient_data.get("diagnosis_type", "Glioblastoma (WHO Grade 4)"),
            "diagnosis_location": patient_data.get("diagnosis_location", "Left temporal lobe"),
            "diagnosis_date": diagnosis_date,
            "next_visit_date": next_visit_date,
            "baseline_kps": int(patient_data.get("baseline_kps", 90)),
            "treatment_stage": patient_data.get("treatment_stage", "Post-Surgical / Adjuvant Stupp Maintenance"),
            "oncologist_name": patient_data.get("oncologist_name", "Dr. Aris Thorne, MD"),
            "nurse_name": patient_data.get("nurse_name", "Sarah Jensen, RN"),
            "share_research_consent": bool(patient_data.get("share_research_consent", True)),
            "created_at": now_iso,
            "phone": patient_data.get("phone", "+1 (555) 234-5678"),
            "email": patient_data.get("email", f"patient.{mrn}@healthmail.org"),
            "pin": str(patient_data.get("pin", "1234"))
        }

        # 1. Insert patient
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("patients").insert(full_record).execute()
            except Exception as e:
                print(f"[DatabaseClient] Supabase patients insert error: {e}")

        # Check if already in storage to avoid duplicate
        idx = next((i for i, p in enumerate(self.storage["patients"]) if p["mrn"] == mrn), None)
        if idx is not None:
            self.storage["patients"][idx] = full_record
        else:
            self.storage["patients"].append(full_record)

        # 2. Initialize Molecular Profile
        mol_record = {
            "profile_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "idh_status": patient_data.get("idh_status", "Mutant"),
            "mgmt_methylated": bool(patient_data.get("mgmt_methylated", True)),
            "codeletion_1p19q": bool(patient_data.get("codeletion_1p19q", False)),
            "tert_promoter_mutant": True,
            "egfr_amplified": False,
            "cdkn2a_deletion": False,
            "created_at": now_iso
        }
        if self.is_connected_to_supabase and self.supabase_client:
            try:
                self.supabase_client.table("molecular_profiles").insert(mol_record).execute()
            except Exception as e:
                print(f"[DatabaseClient] Supabase molecular_profiles insert error: {e}")
        self.storage["molecular_profiles"].append(mol_record)

        # 3. Initialize Baseline Scan
        init_scan = {
            "scan_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "scan_date": today_str,
            "wt_vol_cm3": float(patient_data.get("initial_wt_vol", 12.8)),
            "tc_vol_cm3": float(patient_data.get("initial_tc_vol", 6.5)),
            "et_vol_cm3": float(patient_data.get("initial_et_vol", 3.2)),
            "edema_vol_cm3": float(patient_data.get("initial_edema_vol", 5.4)),
            "dice_score": 0.9120,
            "estimated_rcbv": 1.45,
            "mask_storage_path": f"scans/{mrn}/baseline_seg.nii.gz",
            "created_at": now_iso
        }
        self.storage["mri_scans"].append(init_scan)

        # 4. Initialize Baseline Clinical Report for Patient
        init_report = {
            "report_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "scan_id": init_scan["scan_id"],
            "uploaded_by": "Clinician",
            "report_date": today_str,
            "raw_text": (
                f"EXAMINATION: Baseline multi-parametric MRI examination for {full_record['full_name']} (MRN: {mrn}). "
                f"Evaluation of {full_record['diagnosis_type']} located in {full_record['diagnosis_location']}. "
                f"Whole tumor volume estimated at {init_scan['wt_vol_cm3']:.2f} cm3 with enhancing core measuring {init_scan['et_vol_cm3']:.2f} cm3. "
                f"Surrounding FLAIR hyperintensity consistent with perilesional edema stable at {init_scan['edema_vol_cm3']:.2f} cm3. "
                f"No acute midline shift. DSC perfusion rCBV estimated at {init_scan['estimated_rcbv']:.2f}. "
                f"IMPRESSION: Initial baseline imaging profile established. Follow-up scheduled for {next_visit_date}."
            ),
            "parsed_entities": {
                "tumor_location": full_record["diagnosis_location"],
                "enhancing_dimensions_cm": "1.8 x 2.2",
                "edema_status": f"Stable ({init_scan['edema_vol_cm3']:.2f} cm3)",
                "mass_effect": "Mild / Absent",
                "midline_shift": "None (0 mm)",
                "rcbv": init_scan["estimated_rcbv"],
                "impression": f"Baseline profile for {full_record['diagnosis_type']}"
            },
            "plain_language_summary": (
                f"Welcome to your Care Companion, {full_record['full_name']}! Think of your brain like a wonderful control center. "
                f"Your initial scan gives your care team an exact map of your brain so they can protect it. "
                f"The spot in your {full_record['diagnosis_location']} is clearly mapped, and your medications will act like superhero guards "
                f"to keep healthy areas safe. There is no dangerous pressure, and your next checkup is set for {next_visit_date}."
            ),
            "severity_tag": "Moderate",
            "created_at": now_iso
        }
        self.storage["clinical_reports"].append(init_report)

        # 5. Initialize Baseline Medications with Explicit Scheduled Times
        base_meds = [
            {
                "med_id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "med_name": "Levetiracetam (Keppra)",
                "dosage": patient_data.get("keppra_dosage", "750 mg Oral Tablet"),
                "frequency": "Twice Daily (every 12 hours)",
                "scheduled_time": "08:00 AM & 08:00 PM",
                "instructions": "Seizure prevention. Take consistently every 12 hours with water. Do not skip doses.",
                "prescribed_by": full_record["oncologist_name"],
                "status": "Scheduled",
                "adherence_notes": "Therapeutic seizure prophylaxis",
                "last_updated_by": "Doctor",
                "updated_at": now_iso
            },
            {
                "med_id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "med_name": "Temozolomide (Temodar)",
                "dosage": patient_data.get("tmz_dosage", "150 mg/m² Oral Capsule"),
                "frequency": "Nightly at bedtime (Days 1-5 of 28-day cycle)",
                "scheduled_time": "09:30 PM",
                "instructions": "Chemotherapy. Take on an empty stomach at least 2 hours after dinner with a full glass of water. Premedicate with Ondansetron 30 minutes prior.",
                "prescribed_by": full_record["oncologist_name"],
                "status": "Scheduled",
                "adherence_notes": "Adjuvant Stupp cycle active",
                "last_updated_by": "Doctor",
                "updated_at": now_iso
            },
            {
                "med_id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "med_name": "Ondansetron (Zofran)",
                "dosage": "8 mg Oral Tablet",
                "frequency": "Nightly (30 min prior to Temozolomide)",
                "scheduled_time": "08:30 PM",
                "instructions": "Antiemetic / Nausea prevention. Take 30 to 60 minutes before evening chemotherapy capsule.",
                "prescribed_by": full_record["oncologist_name"],
                "status": "Scheduled",
                "adherence_notes": "Antiemetic coverage",
                "last_updated_by": "Doctor",
                "updated_at": now_iso
            }
        ]
        for bm in base_meds:
            self.storage["medications"].append(bm)

        # 6. Initialize Digital Twin Timeline
        init_timeline = {
            "timeline_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "evaluation_date": today_str,
            "progression_risk_score": 0.2450,
            "triage_category": "Mild",
            "rano_category": "PsP",
            "pseudoprogression_risk": 0.8200,
            "is_pseudoprogression": True,
            "adherence_rate_pct": 100.0,
            "latent_twin_vector": [0.0] * 64,
            "alert_banner_text": f"New Patient Digital Twin initialized for {full_record['full_name']} (MRN: {mrn}). Baseline care plan active.",
            "created_at": now_iso
        }
        self.storage["twin_timeline"].insert(0, init_timeline)

        return full_record



# Global Singleton Client
_global_db_client: Optional[DatabaseClient] = None

def get_database_client(force_reconnect: bool = False) -> DatabaseClient:
    """Returns singleton instance of DatabaseClient, with optional refresh."""
    global _global_db_client
    if _global_db_client is None or force_reconnect:
        _global_db_client = DatabaseClient()
    return _global_db_client

