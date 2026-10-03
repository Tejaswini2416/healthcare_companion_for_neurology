"""
Supabase PostgreSQL Live Synchronization & Telemetry Hub
Enables direct cloud connection management, credential discovery, real-time database health monitoring,
instant one-click remote database seeding, and live telemetry across all digital twin tables.
"""

import streamlit as st
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from ...utils.supabase_client import get_database_client, discover_supabase_credentials


def render_supabase_status_badge():
    """
    Renders a compact, high-visibility status pill in the sidebar.
    Shows live PostgreSQL cloud status vs local resilient engine.
    """
    db = get_database_client()
    status = db.get_connection_status()
    is_live = status["is_connected"]

    badge_bg = "#ecfdf5" if is_live else "#fffbeb"
    border_col = "#10b981" if is_live else "#f59e0b"
    text_col = "#065f46" if is_live else "#92400e"
    indicator_dot = "🟢" if is_live else "🟡"
    status_label = "Supabase PostgreSQL (Live)" if is_live else "Local Resilient Storage"

    st.markdown(
        f"""
        <div style="background: {badge_bg}; border: 1.5px solid {border_col}; border-radius: 8px; padding: 8px 12px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; font-size: 0.82rem; color: {text_col};">
                    {indicator_dot} {status_label}
                </span>
                <span style="font-size: 0.72rem; color: #64748b;">
                    {status['counts'].get('symptom_logs', 0)} Logs
                </span>
            </div>
            <div style="font-size: 0.73rem; color: #475569; margin-top: 3px;">
                {status['url'] if status['url'] else 'In-Memory / Persistent Cache'}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.popover("⚙️ Supabase Settings & Live Sync", use_container_width=True):
        render_supabase_sync_modal()


def render_supabase_sync_modal():
    """Renders the interactive Supabase connection, seeding, and telemetry console."""
    db = get_database_client()
    status = db.get_connection_status()
    is_live = status["is_connected"]

    st.markdown("#### ⚡ Supabase PostgreSQL Connection Hub")
    st.caption(
        "Connect the Patient Digital Twin directly to your Supabase PostgreSQL cloud instance "
        "for real-time multi-device symptom streaming, longitudinal scan tracking, and doctor in-basket alerts."
    )

    # 1. Connection Status Card
    if is_live:
        st.success(
            f"**Connected to Remote Database**: `{status['url']}`\n\n"
            f"• Mode: **Live PostgreSQL Real-Time Sync**\n"
            f"• Telemetry: `{status['counts'].get('patients', 0)}` Patients | "
            f"`{status['counts'].get('mri_scans', 0)}` Scans | "
            f"`{status['counts'].get('symptom_logs', 0)}` Symptoms"
        )
    else:
        st.warning(
            "**Active Mode: Local Resilient Engine (Pre-seeded Demo)**\n\n"
            "Currently operating with high-speed in-memory patient cohorts (MRN 0042, 0043, 0044). "
            "Enter your Supabase credentials below to connect to live PostgreSQL."
        )

    if status.get("connection_error"):
        st.info(f"ℹ️ **Status Notice**: {status['connection_error']}")

    st.divider()

    # 2. Credential Configuration
    st.markdown("##### 🔑 Cloud Database Credentials")
    with st.form("supabase_credentials_form"):
        curr_url = st.session_state.get("custom_supabase_url", status.get("raw_url", ""))
        curr_key = st.session_state.get("custom_supabase_key", "")

        inp_url = st.text_input(
            "Supabase Project URL",
            value=curr_url if curr_url and "your-project" not in curr_url else "",
            placeholder="https://xxxxxxxxxxxxxxxxxxxx.supabase.co",
            help="Found in your Supabase project Dashboard -> Settings -> API -> Project URL"
        )
        inp_key = st.text_input(
            "Supabase API Key (anon public or service_role)",
            value=curr_key,
            type="password",
            placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            help="Found in your Supabase project Dashboard -> Settings -> API -> Project API keys"
        )

        col_save, col_clear = st.columns(2)
        with col_save:
            btn_connect = st.form_submit_button("🔌 Test & Connect to Supabase", type="primary", use_container_width=True)
        with col_clear:
            btn_disconnect = st.form_submit_button("🔄 Revert to Local Storage", use_container_width=True)

        if btn_connect:
            if not inp_url or not inp_key:
                st.error("Please enter both the Supabase Project URL and API Key.")
            else:
                st.session_state.custom_supabase_url = inp_url.strip()
                st.session_state.custom_supabase_key = inp_key.strip()
                ok, msg = db.connect_to_supabase(inp_url.strip(), inp_key.strip())
                if ok:
                    st.toast("Connected to live Supabase PostgreSQL!", icon="🎉")
                    st.success(f"✅ {msg}")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        if btn_disconnect:
            st.session_state.pop("custom_supabase_url", None)
            st.session_state.pop("custom_supabase_key", None)
            db.disconnect_from_supabase()
            st.toast("Switched to Local Resilient Engine", icon="🟡")
            st.rerun()

    # 3. Actions: Database Seeding & Schema Setup
    st.divider()
    st.markdown("##### 🚀 Cloud Database Actions")
    col_seed, col_ping = st.columns(2)

    with col_seed:
        if st.button("🌱 Seed Remote Database", use_container_width=True, help="Populate your remote Supabase PostgreSQL with the full clinical cohort (0042, 0043, 0044)"):
            if not db.is_connected_to_supabase:
                st.warning("Please connect to a live Supabase project first.")
            else:
                with st.spinner("Seeding remote Supabase PostgreSQL tables..."):
                    seed_res = db.seed_remote_supabase()
                    if seed_res["success"]:
                        st.toast("Remote database seeded successfully!", icon="✅")
                        st.success(f"Seeded: {seed_res['seeded_tables']}")
                    else:
                        st.error(f"Seeding completed with notices: {seed_res.get('errors')}")

    with col_ping:
        if st.button("📡 Ping PostgreSQL Latency", use_container_width=True):
            t0 = time.time()
            db.get_patient_by_mrn("0042")
            latency_ms = round((time.time() - t0) * 1000, 1)
            st.toast(f"Round-trip latency: {latency_ms} ms", icon="⚡")
            st.info(f"⏱️ **Database Latency**: `{latency_ms} ms` ({'Supabase Cloud' if db.is_connected_to_supabase else 'Local Memory'})")

    # 4. Schema Helper
    with st.expander("📄 View PostgreSQL Schema SQL (database/supabase_schema.sql)"):
        st.markdown(
            "If your remote Supabase database does not yet have the required tables, "
            "copy and execute the SQL script in your **Supabase Dashboard -> SQL Editor**:"
        )
        st.code(
            """-- Quick setup in Supabase SQL Editor:
-- Run the complete script located in: database/supabase_schema.sql
-- It creates tables: patients, molecular_profiles, mri_scans, clinical_reports,
-- medications, symptom_logs, twin_timeline, counterfactual_simulations
-- and establishes Row-Level-Security (RLS) policies.
""",
            language="sql"
        )
