"""
Scan & Plain-Language Report Detail Component
Provides:
1. Patient Document & External Report Ingestion (uploaded_by='Patient')
2. 2D/3D Multi-Sequence MRI & Sub-Region Tumor Overlay Viewer (NCR, ED, ET, WT)
   with volumetric readouts (WT, TC, ET, ED, Dice confidence, rCBV).
3. Dual-Text Display: Original Technical Radiology Report vs. Patient-Friendly 6th-grade translation.
Supports dynamic patient context across all pre-seeded patients.
"""

import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional

from ...utils.supabase_client import get_database_client
from ...models.mri_segmenter import SyntheticBraTSGenerator
from ...models.clinical_nlp import ClinicalNLPEngine
from .volume_viewer_3d import render_3d_volume_viewer
from .audio_tts_player import render_audio_tts_player


def render_report_viewer():
    """Renders the MRI Volumetric Overlay and Dual-Text Report Viewer."""
    db = get_database_client()
    active_mrn = st.session_state.get("active_patient_mrn", "0042")
    patient = db.get_patient_by_mrn(active_mrn)
    if not patient:
        patient = db.get_patient_by_mrn("0042")
        if not patient:
            st.error("Patient record could not be loaded.")
            return

    patient_id = patient["patient_id"]
    scans = db.get_patient_scans(patient_id)
    reports = db.get_patient_reports(patient_id)

    st.subheader(f"📋 Scans & Plain-Language Reports for {patient['full_name']}")
    st.markdown(
        "Interactive volumetric viewer and plain-language medical translation. "
        "Understand your imaging results without confusing clinical jargon."
    )

    tab_inspect, tab_upload = st.tabs([
        "🔍 Inspect Scans & Translated Reports",
        "📤 Patient Document & Report Upload"
    ])

    # =========================================================================
    # TAB 1: Inspect Scans & Translated Reports
    # =========================================================================
    with tab_inspect:
        if not scans:
            st.warning("No imaging scans recorded for this patient yet.")
            return

        # Select scan date to inspect
        scan_options = {f"{s['scan_date']} (WT: {s['wt_vol_cm3']:.1f} cm³, Dice: {s['dice_score']:.3f})": s for s in scans}
        selected_label = st.selectbox(
            "Select MRI Scan Evaluation Date:",
            options=list(scan_options.keys()),
            index=len(scan_options)-1
        )
        current_scan = scan_options[selected_label]
        matching_report = next((r for r in reports if r.get("scan_id") == current_scan.get("scan_id")), reports[0] if reports else None)

        st.write("")

        # Top Volumetric Metric Cards
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Whole Tumor (WT)", f"{current_scan['wt_vol_cm3']:.1f} cm³")
        with m2:
            st.metric("Tumor Core (TC)", f"{current_scan['tc_vol_cm3']:.1f} cm³")
        with m3:
            st.metric("Enhancing (ET)", f"{current_scan['et_vol_cm3']:.1f} cm³")
        with m4:
            st.metric("Edema (ED)", f"{current_scan['edema_vol_cm3']:.1f} cm³")
        with m5:
            st.metric("SegResNet Dice", f"{current_scan['dice_score']:.4f}")
        with m6:
            st.metric("Perfusion rCBV", f"{current_scan['estimated_rcbv']:.2f}", help="<1.75 indicates hypoperfusion / radiation necrosis")

        st.divider()

        # Side-by-Side: Left = MRI 3D/2D Viewer, Right = Multilingual Dual-Text Report with Audio TTS
        col_img, col_report = st.columns([1.1, 0.9])

        with col_img:
            st.markdown("##### 🧠 3D MRI Volume Reconstruction & Multi-Sequence Slices")

            tab_3d, tab_2d = st.tabs([
                "🌐 Interactive 3D WebGL Volume",
                "🔬 2D Multi-Sequence Slices"
            ])

            with tab_3d:
                render_3d_volume_viewer(
                    scan_data=current_scan,
                    patient_name=patient['full_name'],
                    mrn=patient['mrn'],
                    height=480
                )

            with tab_2d:
                # Cache or generate 3D volume for slice viewing
                cache_key = f"cached_mri_{patient['mrn']}"
                if cache_key not in st.session_state:
                    vol_4ch, seg_gt = SyntheticBraTSGenerator.generate_synthetic_volume(
                        shape=(64, 64, 48),
                        wt_radius=13.0,
                        tc_radius=8.0,
                        et_thickness=3.6
                    )
                    st.session_state[cache_key] = (vol_4ch, seg_gt)

                mri_vol, seg_gt = st.session_state[cache_key]

                c_seq, c_mask, c_slice = st.columns([1, 1, 1])
                with c_seq:
                    seq_choice = st.selectbox("Sequence", ["T1ce (Contrast)", "FLAIR (Edema)", "T2", "T1"], index=0)
                with c_mask:
                    mask_choice = st.selectbox("Overlay", ["Sub-Regions (Color)", "Whole Tumor (WT)", "Tumor Core (TC)", "None"], index=0)
                with c_slice:
                    slice_idx = st.slider("Axial Slice Z Index", 10, 40, 24)

                # Map sequence to channel
                seq_idx = {"T1": 0, "T1ce (Contrast)": 1, "T2": 2, "FLAIR (Edema)": 3}[seq_choice]
                base_slice = mri_vol[seq_idx, :, :, slice_idx]

                # Sub-region masks
                ncr_slice = seg_gt[0, :, :, slice_idx]
                ed_slice = seg_gt[1, :, :, slice_idx]
                et_slice = seg_gt[2, :, :, slice_idx]

                # Matplotlib visualization
                fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#0f172a")
                ax.imshow(base_slice, cmap="gray", origin="lower", vmin=0, vmax=1.0)

                if mask_choice == "Sub-Regions (Color)":
                    overlay = np.zeros((*base_slice.shape, 4), dtype=np.float32)
                    overlay[ncr_slice > 0] = [0.9, 0.1, 0.1, 0.55] # Red: Necrosis
                    overlay[ed_slice > 0]  = [0.1, 0.8, 0.2, 0.40] # Green: Edema
                    overlay[et_slice > 0]  = [0.0, 0.8, 0.9, 0.65] # Cyan: Enhancing
                    ax.imshow(overlay, origin="lower")
                elif mask_choice == "Whole Tumor (WT)":
                    wt_slice = np.clip(ncr_slice | ed_slice | et_slice, 0, 1)
                    overlay = np.zeros((*base_slice.shape, 4), dtype=np.float32)
                    overlay[wt_slice > 0] = [0.1, 0.8, 0.3, 0.50]
                    ax.imshow(overlay, origin="lower")
                elif mask_choice == "Tumor Core (TC)":
                    tc_slice = np.clip(ncr_slice | et_slice, 0, 1)
                    overlay = np.zeros((*base_slice.shape, 4), dtype=np.float32)
                    overlay[tc_slice > 0] = [0.9, 0.2, 0.2, 0.60]
                    ax.imshow(overlay, origin="lower")

                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption(
                    "🎨 **Legend**: 🔴 Red = Necrotic debris • 🟢 Green = Peritumoral Edema • 🔵 Cyan = Active Enhancing Rim"
                )

        with col_report:
            st.markdown("##### 📄 Side-by-Side Dual Radiology Report Translator")

            if matching_report:
                # Medical Terminology Popovers
                col_pop1, col_pop2, col_pop3 = st.columns(3)
                with col_pop1:
                    with st.popover("💡 What is 'rCBV'?"):
                        st.markdown(
                            "**Relative Cerebral Blood Volume (rCBV)**:\n\n"
                            "Measures microvascular blood flow inside the treated lesion. "
                            "Values **< 1.75** indicate hypoperfusion / radiation necrosis (benign healing). "
                            "Values **> 1.75** warrant closer evaluation for active tumor growth."
                        )
                with col_pop2:
                    with st.popover("💡 What is 'PsP'?"):
                        st.markdown(
                            "**Pseudoprogression (PsP)**:\n\n"
                            "A positive treatment response where radiation and chemotherapy cause inflammatory "
                            "breakdown of tumor cells. It can mimic tumor enlargement on scans but is actually "
                            "a sign that therapy is working."
                        )
                with col_pop3:
                    with st.popover("💡 'MGMT Status'?"):
                        st.markdown(
                            "**MGMT Promoter Methylation**:\n\n"
                            "A favorable biomarker indicating high sensitivity to alkylating chemotherapy (Temozolomide). "
                            "Patients with methylated MGMT have significantly higher odds of pseudoprogression and long-term control."
                        )

                # Multilingual Selection
                lang_map = {
                    "English 🇺🇸": ("en", "en-US"),
                    "Español (Spanish) 🇪🇸": ("es", "es-ES"),
                    "हिंदी (Hindi) 🇮🇳": ("hi", "hi-IN"),
                    "中文 (Mandarin) 🇨🇳": ("zh", "zh-CN"),
                    "Français (French) 🇫🇷": ("fr", "fr-FR")
                }

                col_l_sel, col_l_badge = st.columns([2.2, 1.2])
                with col_l_sel:
                    selected_lang_label = st.selectbox(
                        "🌐 Select Translation Language:",
                        options=list(lang_map.keys()),
                        index=0,
                        key="report_viewer_lang_selector"
                    )
                with col_l_badge:
                    st.caption("Accessibility Mode Active")

                lang_code, speech_locale = lang_map[selected_lang_label]

                # Resolve multi-language text
                report_translations = matching_report.get("translations", {})
                if lang_code in report_translations:
                    active_summary = report_translations[lang_code]
                else:
                    nlp_engine = ClinicalNLPEngine(offline=True)
                    active_summary = nlp_engine.translate_to_plain_language(
                        matching_report['raw_text'],
                        matching_report.get('parsed_entities'),
                        language=lang_code
                    )

                # Side-by-Side Dual Display
                col_tech, col_plain = st.columns(2)
                with col_tech:
                    st.markdown("###### 🔬 Technical Radiology Report Excerpt")
                    st.info(matching_report['raw_text'])
                with col_plain:
                    st.markdown(f"###### 💚 Empathetic 6th-Grade Summary ({selected_lang_label.split()[0]})")
                    st.success(active_summary)

                # Audio Text-to-Speech Accessibility Player
                render_audio_tts_player(
                    text_to_speak=active_summary,
                    language_code=speech_locale,
                    label=f"Listen to Summary ({selected_lang_label.split()[0]})",
                    height=100
                )

                st.markdown("###### 🔍 Extracted Clinical Entities")
                parsed = matching_report.get('parsed_entities', {})
                if parsed:
                    c_e1, c_e2 = st.columns(2)
                    with c_e1:
                        st.markdown(f"• **Location**: `{parsed.get('tumor_location', 'N/A')}`")
                        st.markdown(f"• **Dimensions**: `{parsed.get('enhancing_dimensions_cm', 'N/A')} cm`")
                    with c_e2:
                        st.markdown(f"• **Perfusion rCBV**: `{parsed.get('rcbv', 'N/A')}`")
                        st.markdown(f"• **Severity Level**: `{matching_report.get('severity_tag', 'N/A')}`")
            else:
                st.info("No text report available for this scan date.")

    # =========================================================================
    # TAB 2: Patient Document & Report Upload
    # =========================================================================
    with tab_upload:
        st.markdown("##### 📤 Upload External Records, Lab PDFs, or Doctor Notes")
        st.markdown(
            "Have you received a consultation note, second opinion, or pathology report from outside our clinic? "
            "Upload or paste it here to automatically translate it into plain language and update your Patient Digital Twin."
        )

        up_pdf = st.file_uploader("Upload Medical Document (.txt, .pdf)", type=["txt", "pdf"], key="patient_doc_up")
        pasted_text = st.text_area(
            "Or Paste Medical Notes / Findings Directly:",
            placeholder="e.g. Brain MRI scan showing interval stability in the left temporal lobe. No midline shift. Minimal residual edema...",
            height=130,
            key="patient_pasted_report"
        )

        if st.button("🚀 Process & Translate Document", type="primary", use_container_width=True):
            input_content = pasted_text.strip()
            if not input_content and up_pdf:
                input_content = up_pdf.read().decode("utf-8", errors="replace")

            if not input_content:
                st.error("Please paste report text or upload a document to analyze.")
            else:
                with st.spinner("Analyzing medical findings with Bio_ClinicalBERT and generating patient-friendly summary..."):
                    nlp_engine = ClinicalNLPEngine(offline=True)
                    nlp_res = nlp_engine.analyze_report(input_content)

                    # Save to database
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    db.insert_clinical_report({
                        "patient_id": patient_id,
                        "scan_id": current_scan.get("scan_id"),
                        "uploaded_by": "Patient",
                        "report_date": today_str,
                        "raw_text": input_content,
                        "parsed_entities": nlp_res["parsed_entities"],
                        "plain_language_summary": nlp_res["plain_language_summary"],
                        "severity_tag": nlp_res["severity_tag"]
                    })

                st.success("✅ Document processed successfully! Translated summary added to your records.")
                st.markdown("##### 🌟 Translated Plain-Language Summary:")
                st.info(nlp_res["plain_language_summary"])
                st.rerun()
