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
from ...models.clinical_nlp import ClinicalNLPEngine, extract_text_from_file
from ...utils.pdf_generator import generate_patient_clinical_pdf
from .brain_3d_viewer import render_3d_brain_viewer_panel
from .audio_tts_player import render_audio_tts_player
from .report_verification_hub import validate_radiology_or_pathology_report


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

    if not scans:
        st.warning("No imaging scans recorded for this patient yet.")
        return

    # Select scan date to inspect
    scan_options = {f"{s['scan_date']} (WT: {s['wt_vol_cm3']:.1f} cm³, Dice: {s['dice_score']:.3f})": s for s in scans}
    selected_label = st.selectbox(
        "Select MRI Scan Evaluation Date:",
        options=list(scan_options.keys()),
        index=len(scan_options)-1,
        key="report_viewer_scan_date_sel"
    )
    current_scan = scan_options[selected_label]
    matching_report = next((r for r in reports if r.get("scan_id") == current_scan.get("scan_id")), reports[0] if reports else None)

    tab_inspect, tab_3d_brain, tab_upload = st.tabs([
        "🔍 Inspect Scans & Translated Reports",
        "🧠 3D Brain & Tumor Localization",
        "📤 Patient Document & Report Upload"
    ])

    # =========================================================================
    # TAB 1: Inspect Scans & Translated Reports
    # =========================================================================
    with tab_inspect:
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
            st.markdown("##### 🧠 MRI Multi-Sequence Slices & Pathology Overlays")

            # Dynamic MRI volumetric slice cache keyed by patient and scan ID
            scan_key_id = current_scan.get("scan_id", current_scan.get("scan_date", "latest"))
            cache_key = f"cached_mri_{patient['mrn']}_{scan_key_id}"
            if cache_key not in st.session_state:
                wt_v = float(current_scan.get("wt_vol_cm3", 14.6))
                tc_v = float(current_scan.get("tc_vol_cm3", 5.0))
                et_v = float(current_scan.get("et_vol_cm3", 3.8))
                wt_radius = min(22.0, max(6.0, float(np.cbrt(wt_v * 5.5))))
                tc_radius = min(17.0, max(4.0, float(np.cbrt(tc_v * 4.8))))
                et_thickness = min(5.0, max(1.8, et_v * 0.85))
                vol_4ch, seg_gt = SyntheticBraTSGenerator.generate_synthetic_volume(
                    shape=(64, 64, 48),
                    wt_radius=wt_radius,
                    tc_radius=tc_radius,
                    et_thickness=et_thickness
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

            # Report selector across all reports on file
            if reports:
                if len(reports) > 1:
                    rep_options = {
                        f"📄 {r.get('report_date', 'Recent')} • [{r.get('severity_tag', 'Report')}] • {r.get('uploaded_by', 'Clinician')}": r
                        for r in reports
                    }
                    sel_rep_key = st.selectbox(
                        "Select Clinical Report on File:",
                        options=list(rep_options.keys()),
                        index=0,
                        key="report_viewer_report_selector_dropdown"
                    )
                    matching_report = rep_options[sel_rep_key]

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
                elif lang_code == "en" and matching_report.get("plain_language_summary"):
                    active_summary = matching_report["plain_language_summary"]
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
                    st.markdown("###### 💚 Empathetic Plain-Language Summary")
                    st.success(active_summary)

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

                st.write("")
                # Official PDF Download Button for Patient
                pdf_bytes = generate_patient_clinical_pdf(
                    patient=patient,
                    report=matching_report,
                    scan_data=current_scan,
                    molecular_data=db.get_patient_molecular(patient_id)
                )
                st.download_button(
                    label="📥 Download Official Clinical Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"Clinical_Report_{patient['mrn']}_{matching_report.get('report_date', 'date')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            else:
                st.info("No text report available for this scan date.")

    # =========================================================================
    # TAB 2: Interactive 3D Brain & Tumor Localization
    # =========================================================================
    with tab_3d_brain:
        render_3d_brain_viewer_panel(
            patient=patient,
            scan_data=current_scan,
            panel_title=f"🧠 Interactive 3D Brain Reconstruction for {patient['full_name']}",
            key_prefix="patient_report_viewer_3d"
        )

    # =========================================================================
    # TAB 3: Patient Document & Report Upload
    # =========================================================================
    with tab_upload:
        st.markdown("##### 📤 Upload External Records, Lab PDFs, or Doctor Notes")
        st.markdown(
            "Upload or paste an external neuro-radiology or surgical pathology report here to automatically extract findings, "
            "translate them into plain language, and update your Patient Digital Twin. "
            "*(Only authentic clinical Radiology or Pathology reports are accepted.)*"
        )

        up_pdf = st.file_uploader(
            "Upload Medical Document (.pdf, .docx, .txt, .md, .json)",
            type=["pdf", "docx", "txt", "md", "json"],
            key="patient_doc_up",
            help="Drop Radiology (MRI, CT) or Pathology reports here."
        )

        extracted_text = ""
        if up_pdf is not None:
            raw_up_text = extract_text_from_file(up_pdf)
            if raw_up_text:
                is_valid_rep, rep_val_msg = validate_radiology_or_pathology_report(raw_up_text)
                if not is_valid_rep:
                    st.error(rep_val_msg)
                else:
                    extracted_text = raw_up_text
                    st.success(f"✅ {rep_val_msg} Extracted **{len(extracted_text.split())}** words from `{up_pdf.name}`.")

        pasted_text = st.text_area(
            "Medical Notes / Findings (Extracted or Pasted):",
            value=extracted_text if extracted_text else "",
            placeholder="e.g. Brain MRI scan showing interval stability in the left temporal lobe. Residual enhancing lesion 1.4 x 1.8 cm. No midline shift. Minimal residual edema...",
            height=150,
            key="patient_pasted_report"
        )

        if st.button("🚀 Process & Analyze Document", type="primary", use_container_width=True):
            input_content = pasted_text.strip() or extracted_text.strip()

            if not input_content:
                st.error("Please paste report text or upload a document to analyze.")
            else:
                is_valid_doc, doc_msg = validate_radiology_or_pathology_report(input_content)
                if not is_valid_doc:
                    st.error(doc_msg)
                else:
                    with st.spinner("Analyzing medical findings with Bio_ClinicalBERT and generating patient-friendly summary..."):
                        nlp_engine = ClinicalNLPEngine(offline=True)
                        nlp_res = nlp_engine.analyze_report(input_content)

                    # Save to database
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    db.insert_clinical_report({
                        "patient_id": patient_id,
                        "scan_id": current_scan.get("scan_id") if 'current_scan' in locals() else None,
                        "uploaded_by": "Patient",
                        "report_date": today_str,
                        "raw_text": input_content,
                        "parsed_entities": nlp_res["parsed_entities"],
                        "plain_language_summary": nlp_res["plain_language_summary"],
                        "translations": nlp_res.get("translations", {}),
                        "severity_tag": nlp_res["severity_tag"]
                    })
                    st.session_state.last_uploaded_analysis = nlp_res

                st.success("✅ Clinical report successfully analyzed and saved to your health record!")

        # Display immediate analysis if available
        last_analysis = st.session_state.get("last_uploaded_analysis")
        if last_analysis:
            st.markdown("---")
            st.markdown("##### 🌟 AI Analysis & Translation Results")

            ent = last_analysis.get("parsed_entities", {})
            c_r1, c_r2, c_r3, c_r4 = st.columns(4)
            with c_r1:
                st.metric("Clinical Severity", last_analysis.get("severity_tag", "Mild"))
            with c_r2:
                st.metric("Location", ent.get("tumor_location", "N/A"))
            with c_r3:
                st.metric("Dimensions", f"{ent.get('enhancing_dimensions_cm', 'N/A')} cm")
            with c_r4:
                st.metric("Perfusion rCBV", f"{ent.get('rcbv', 1.50):.2f}")

            st.markdown("###### 💚 Empathetic Plain-Language Summary:")
            st.info(last_analysis["plain_language_summary"])
