-- ============================================================================
-- HEALTHCARE COMPANION: AI-POWERED CANCER CARE COMPANION (PATIENT DIGITAL TWIN FOR GLIOMA)
-- Database: Supabase / PostgreSQL Schema & Row Level Security (RLS)
-- File: database/supabase_schema.sql
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. TABLE: patients
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.patients (
    patient_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mrn TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    age INT NOT NULL CHECK (age > 0 AND age < 130),
    gender TEXT NOT NULL,
    diagnosis_type TEXT NOT NULL, -- e.g. 'Glioblastoma, IDH-wildtype', 'Astrocytoma, IDH-mutant'
    diagnosis_location TEXT NOT NULL, -- e.g. 'Left Temporal Lobe', 'Right Frontal Lobe'
    diagnosis_date DATE NOT NULL,
    treatment_stage TEXT NOT NULL, -- e.g. 'Adjuvant Chemoradiation Cycle 3'
    baseline_kps INT NOT NULL CHECK (baseline_kps BETWEEN 10 AND 100),
    oncologist_name TEXT NOT NULL,
    nurse_name TEXT NOT NULL,
    share_research_consent BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_patients_mrn ON public.patients(mrn);

-- ============================================================================
-- 2. TABLE: molecular_profiles
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.molecular_profiles (
    profile_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    idh_status TEXT NOT NULL CHECK (idh_status IN ('Mutant', 'Wild-Type')),
    mgmt_methylated BOOLEAN NOT NULL,
    codeletion_1p19q BOOLEAN NOT NULL,
    assayed_at DATE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_molecular_patient ON public.molecular_profiles(patient_id);

-- ============================================================================
-- 3. TABLE: mri_scans
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.mri_scans (
    scan_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    scan_date DATE NOT NULL,
    wt_vol_cm3 NUMERIC(8, 2) NOT NULL, -- Whole Tumor (NCR + ED + ET)
    tc_vol_cm3 NUMERIC(8, 2) NOT NULL, -- Tumor Core (NCR + ET)
    et_vol_cm3 NUMERIC(8, 2) NOT NULL, -- Enhancing Tumor
    edema_vol_cm3 NUMERIC(8, 2) NOT NULL, -- Peritumoral Edema
    dice_score NUMERIC(5, 4) NOT NULL, -- Segmentation confidence / Dice
    estimated_rcbv NUMERIC(5, 2) NOT NULL, -- Relative Cerebral Blood Volume
    mask_storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_mri_patient_date ON public.mri_scans(patient_id, scan_date DESC);

-- ============================================================================
-- 4. TABLE: clinical_reports
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.clinical_reports (
    report_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    scan_id UUID REFERENCES public.mri_scans(scan_id) ON DELETE SET NULL,
    uploaded_by TEXT NOT NULL DEFAULT 'Clinician' CHECK (uploaded_by IN ('Clinician', 'Patient')),
    report_date DATE NOT NULL,
    raw_text TEXT NOT NULL,
    parsed_entities JSONB NOT NULL,
    plain_language_summary TEXT NOT NULL,
    severity_tag TEXT NOT NULL CHECK (severity_tag IN ('Mild', 'Moderate', 'Critical')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_reports_patient ON public.clinical_reports(patient_id);

-- ============================================================================
-- 5. TABLE: medications
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.medications (
    med_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    med_name TEXT NOT NULL,
    dosage TEXT NOT NULL,
    frequency TEXT NOT NULL,
    instructions TEXT,
    prescribed_by TEXT NOT NULL DEFAULT 'Oncology Team',
    status TEXT NOT NULL DEFAULT 'Due' CHECK (status IN ('Taken', 'Due', 'Missed')),
    adherence_notes TEXT,
    last_updated_by TEXT NOT NULL DEFAULT 'Doctor' CHECK (last_updated_by IN ('Doctor', 'Patient')),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_meds_patient ON public.medications(patient_id);

-- ============================================================================
-- 6. TABLE: symptom_logs
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.symptom_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    logged_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    symptom_type TEXT NOT NULL CHECK (symptom_type IN ('Headache', 'Vision', 'Seizure', 'Fatigue', 'Speech', 'Motor')),
    severity TEXT NOT NULL CHECK (severity IN ('Mild', 'Moderate', 'Severe')),
    onset_time TEXT NOT NULL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_symptoms_patient_time ON public.symptom_logs(patient_id, logged_at DESC);

-- ============================================================================
-- 7. TABLE: twin_timeline
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.twin_timeline (
    timeline_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    evaluation_date DATE NOT NULL,
    progression_risk_score NUMERIC(5, 4) NOT NULL, -- 0.0000 - 1.0000
    triage_category TEXT NOT NULL CHECK (triage_category IN ('Mild', 'Moderate', 'Critical')),
    rano_category TEXT NOT NULL CHECK (rano_category IN ('CR', 'PR', 'SD', 'PD', 'PsP')),
    pseudoprogression_risk NUMERIC(5, 4) NOT NULL,
    is_pseudoprogression BOOLEAN NOT NULL DEFAULT false,
    adherence_rate_pct NUMERIC(5, 2) NOT NULL,
    latent_twin_vector JSONB NOT NULL, -- 64-dim vector
    alert_banner_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_twin_patient_eval ON public.twin_timeline(patient_id, evaluation_date DESC);

-- ============================================================================
-- 8. TABLE: counterfactual_simulations
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.counterfactual_simulations (
    sim_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    simulation_name TEXT NOT NULL,
    intervention_type TEXT NOT NULL CHECK (intervention_type IN ('SOC_TMZ', 'DOSE_DENSE', 'HOLD_TMZ', 'LOMUSTINE')),
    forecast_horizon_days INT NOT NULL,
    predicted_trajectory JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_sim_patient ON public.counterfactual_simulations(patient_id);

-- ============================================================================
-- 9. ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.molecular_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mri_scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.clinical_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.medications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.symptom_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.twin_timeline ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.counterfactual_simulations ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE
    t text;
BEGIN
    FOR t IN 
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
          AND table_name IN ('patients', 'molecular_profiles', 'mri_scans', 'clinical_reports', 
                             'medications', 'symptom_logs', 'twin_timeline', 'counterfactual_simulations')
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS "Allow all for authenticated and anon" ON public.%I;', t);
        EXECUTE format('CREATE POLICY "Allow all for authenticated and anon" ON public.%I FOR ALL TO authenticated, anon USING (true) WITH CHECK (true);', t);
    END LOOP;
END $$;

-- ============================================================================
-- 10. PRE-SEEDED SEED DATA: MULTIPLE REALISTIC PATIENT PROFILES
-- ============================================================================

-- PATIENT 1: V. Thanuja (MRN: 0042)
INSERT INTO public.patients (
    patient_id, mrn, full_name, age, gender, diagnosis_type, diagnosis_location,
    diagnosis_date, treatment_stage, baseline_kps, oncologist_name, nurse_name, share_research_consent
) VALUES (
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00042',
    '0042',
    'V. Thanuja',
    24,
    'Female',
    'High-Grade Glioma (Astrocytoma, IDH-Mutant Grade 4)',
    'Left Temporal Lobe',
    '2026-05-14',
    'Adjuvant Chemoradiation (Stupp Protocol Cycle 3)',
    90,
    'Dr. Aris Thorne, MD (Neuro-Oncology)',
    'Sarah Jensen, RN (Care Navigator)',
    true
) ON CONFLICT (mrn) DO UPDATE SET full_name = EXCLUDED.full_name;

INSERT INTO public.molecular_profiles (
    profile_id, patient_id, idh_status, mgmt_methylated, codeletion_1p19q, assayed_at
) VALUES (
    'f1b38d38-2c26-4d2b-91c6-2c1b97b00001',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00042',
    'Mutant', true, false, '2026-05-20'
) ON CONFLICT DO NOTHING;

INSERT INTO public.mri_scans (
    scan_id, patient_id, scan_date, wt_vol_cm3, tc_vol_cm3, et_vol_cm3, edema_vol_cm3, dice_score, estimated_rcbv, mask_storage_path
) VALUES 
('a1b38d38-2c26-4d2b-91c6-2c1b97b00011', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', '2026-06-10', 22.40, 14.80, 8.20, 7.60, 0.9120, 2.15, 'scans/0042/20260610_seg.nii.gz'),
('a2b38d38-2c26-4d2b-91c6-2c1b97b00012', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', '2026-07-28', 17.10, 10.40, 5.60, 6.70, 0.9340, 1.82, 'scans/0042/20260728_seg.nii.gz'),
('a3b38d38-2c26-4d2b-91c6-2c1b97b00013', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', '2026-09-18', 14.60, 8.10, 4.10, 6.50, 0.9410, 1.48, 'scans/0042/20260918_seg.nii.gz')
ON CONFLICT DO NOTHING;

INSERT INTO public.clinical_reports (
    report_id, patient_id, scan_id, uploaded_by, report_date, raw_text, parsed_entities, plain_language_summary, severity_tag
) VALUES (
    'b1b38d38-2c26-4d2b-91c6-2c1b97b00021',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00042',
    'a3b38d38-2c26-4d2b-91c6-2c1b97b00013',
    'Clinician',
    '2026-09-19',
    'FINDINGS: Interval reduction in enhancing residual lesion along posterior resection margin in left temporal lobe, measuring 1.4 x 1.8 cm. Enhancing tumor volume estimated at 4.10 cm3. Moderate FLAIR hyperintensity consistent with perilesional vasogenic edema stable at 6.50 cm3. No significant mass effect or midline shift. Ventricles are symmetric. Relative CBV is reduced at 1.48. IMPRESSION: Consistent with post-treatment radiation effect / pseudoprogression.',
    '{"tumor_location": "Left temporal lobe (resection margin)", "enhancing_dimensions_cm": "1.4 x 1.8", "edema_status": "Stable (6.50 cm3)", "mass_effect": false, "midline_shift_mm": 0.0, "rcbv": 1.48, "impression": "Treatment-induced radiation necrosis / pseudoprogression"}'::jsonb,
    'Your latest brain scan shows encouraging progress! The active area of your treated tumor has decreased in size to 4.1 cubic centimeters. The mild tissue swelling (edema) around the surgery area is completely stable and not pushing against any healthy brain regions. The low blood flow measurement indicates that the changes seen are safe, expected healing reactions from your radiation therapy (pseudoprogression), rather than active cancer growth.',
    'Moderate'
) ON CONFLICT DO NOTHING;

INSERT INTO public.medications (
    med_id, patient_id, med_name, dosage, frequency, instructions, prescribed_by, status, adherence_notes, last_updated_by
) VALUES 
('c1b38d38-2c26-4d2b-91c6-2c1b97b00031', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', 'Temozolomide (Temodar)', '150 mg/m2 Oral Capsule', 'Nightly (Days 1-5 of 28-day cycle)', 'Take on an empty stomach with antiemetic 30 minutes prior', 'Dr. Aris Thorne, MD', 'Taken', 'Cycle 3 Day 4 completed smoothly', 'Doctor'),
('c2b38d38-2c26-4d2b-91c6-2c1b97b00032', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', 'Levetiracetam (Keppra)', '750 mg Oral Tablet', 'Twice Daily (Every 12 hours)', 'Strict seizure prophylaxis; do not abruptly discontinue', 'Dr. Aris Thorne, MD', 'Missed', 'Patient reported morning nausea and missed morning dose', 'Patient'),
('c3b38d38-2c26-4d2b-91c6-2c1b97b00033', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', 'Dexamethasone', '2 mg Oral Tablet', 'Once Daily (Morning with food)', 'Steroid taper for vasogenic edema control', 'Dr. Aris Thorne, MD', 'Taken', 'Dose reduced per schedule', 'Doctor')
ON CONFLICT DO NOTHING;

INSERT INTO public.symptom_logs (
    log_id, patient_id, logged_at, symptom_type, severity, onset_time, notes
) VALUES 
('d1b38d38-2c26-4d2b-91c6-2c1b97b00041', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', timezone('utc'::text, now() - INTERVAL '2 days'), 'Fatigue', 'Moderate', 'Late afternoon', 'Needed a 2 hour nap after taking temozolomide.'),
('d2b38d38-2c26-4d2b-91c6-2c1b97b00042', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', timezone('utc'::text, now() - INTERVAL '1 day'), 'Seizure', 'Moderate', 'Morning 9:30 AM', 'Felt strange olfactory sensations (burning smell) and mild right hand tingling for 45 seconds after missing morning Keppra.'),
('d3b38d38-2c26-4d2b-91c6-2c1b97b00043', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00042', timezone('utc'::text, now() - INTERVAL '4 hours'), 'Headache', 'Mild', 'Morning 7:00 AM', 'Dull throbbing in left temple, relieved by drinking water.')
ON CONFLICT DO NOTHING;

INSERT INTO public.twin_timeline (
    timeline_id, patient_id, evaluation_date, progression_risk_score, triage_category, rano_category, pseudoprogression_risk, is_pseudoprogression, adherence_rate_pct, latent_twin_vector, alert_banner_text
) VALUES (
    'e1b38d38-2c26-4d2b-91c6-2c1b97b00051',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00042',
    '2026-09-20',
    0.2850,
    'Moderate',
    'PsP',
    0.8420,
    true,
    85.70,
    '{"latent_dim": 64, "modality_weights": {"mri": 0.42, "nlp": 0.31, "clinical": 0.27}}'::jsonb,
    'Attention: Missed Levetiracetam (Keppra) dose coincides with a newly reported sensory seizure aura. Please take your prescribed dose and contact your care team.'
) ON CONFLICT DO NOTHING;

-- PATIENT 2: Marcus Chen (MRN: 0043)
INSERT INTO public.patients (
    patient_id, mrn, full_name, age, gender, diagnosis_type, diagnosis_location,
    diagnosis_date, treatment_stage, baseline_kps, oncologist_name, nurse_name, share_research_consent
) VALUES (
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00043',
    '0043',
    'Marcus Chen',
    58,
    'Male',
    'Glioblastoma (IDH-Wildtype WHO Grade 4)',
    'Right Frontal Lobe',
    '2026-07-02',
    'Post-Surgical Adjuvant Temozolomide Cycle 1',
    80,
    'Dr. Elena Rostova, MD (Neuro-Oncology)',
    'James Miller, RN (Care Navigator)',
    true
) ON CONFLICT (mrn) DO UPDATE SET full_name = EXCLUDED.full_name;

INSERT INTO public.molecular_profiles (
    profile_id, patient_id, idh_status, mgmt_methylated, codeletion_1p19q, assayed_at
) VALUES (
    'f1b38d38-2c26-4d2b-91c6-2c1b97b00002',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00043',
    'Wild-Type', false, false, '2026-07-10'
) ON CONFLICT DO NOTHING;

INSERT INTO public.mri_scans (
    scan_id, patient_id, scan_date, wt_vol_cm3, tc_vol_cm3, et_vol_cm3, edema_vol_cm3, dice_score, estimated_rcbv, mask_storage_path
) VALUES 
('a4b38d38-2c26-4d2b-91c6-2c1b97b00014', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00043', '2026-07-05', 38.60, 26.20, 15.80, 12.40, 0.9050, 3.40, 'scans/0043/20260705_seg.nii.gz'),
('a5b38d38-2c26-4d2b-91c6-2c1b97b00015', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00043', '2026-09-02', 29.40, 18.10, 10.90, 11.30, 0.9280, 2.75, 'scans/0043/20260902_seg.nii.gz')
ON CONFLICT DO NOTHING;

INSERT INTO public.clinical_reports (
    report_id, patient_id, scan_id, uploaded_by, report_date, raw_text, parsed_entities, plain_language_summary, severity_tag
) VALUES (
    'b2b38d38-2c26-4d2b-91c6-2c1b97b00022',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00043',
    'a5b38d38-2c26-4d2b-91c6-2c1b97b00015',
    'Clinician',
    '2026-09-03',
    'FINDINGS: Post-resection cavity in right frontal lobe with residual peripheral enhancement measuring 2.1 x 2.4 cm. Mild reduction in overall volume. Surrounding FLAIR hyperintensity stable. No midline shift. Perfusion rCBV elevated at 2.75. IMPRESSION: Consistent with residual active tumor responding partially to early post-operative adjuvant therapy.',
    '{"tumor_location": "Right frontal lobe", "enhancing_dimensions_cm": "2.1 x 2.4", "edema_status": "Stable (11.30 cm3)", "mass_effect": false, "midline_shift_mm": 0.0, "rcbv": 2.75, "impression": "Residual active high-grade tumor"}'::jsonb,
    'Your recent scan shows that the surgery was effective and your treatment is stabilizing the area. The remaining edge of the tumor in the right front portion of your brain has gotten smaller, down to 10.9 cubic centimeters. We will continue active monitoring and your scheduled chemotherapy cycle.',
    'Moderate'
) ON CONFLICT DO NOTHING;

INSERT INTO public.medications (
    med_id, patient_id, med_name, dosage, frequency, instructions, prescribed_by, status, adherence_notes, last_updated_by
) VALUES 
('c4b38d38-2c26-4d2b-91c6-2c1b97b00034', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00043', 'Temozolomide (Temodar)', '200 mg/m2 Oral Capsule', 'Nightly (Days 1-5 of 28-day cycle)', 'Take with antiemetic on empty stomach', 'Dr. Elena Rostova, MD', 'Taken', 'Completed cycle days 1-3 on schedule', 'Doctor'),
('c5b38d38-2c26-4d2b-91c6-2c1b97b00035', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00043', 'Levetiracetam (Keppra)', '1000 mg Oral Tablet', 'Twice Daily (Every 12 hours)', 'Seizure prevention; do not miss doses', 'Dr. Elena Rostova, MD', 'Taken', 'Consistently taken with meals', 'Patient'),
('c6b38d38-2c26-4d2b-91c6-2c1b97b00036', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00043', 'Ondansetron (Zofran)', '8 mg Oral Tablet', 'As needed 30 min before TMZ', 'For chemotherapy-induced nausea', 'Dr. Elena Rostova, MD', 'Taken', 'Took before evening capsule', 'Patient')
ON CONFLICT DO NOTHING;

INSERT INTO public.symptom_logs (
    log_id, patient_id, logged_at, symptom_type, severity, onset_time, notes
) VALUES 
('d4b38d38-2c26-4d2b-91c6-2c1b97b00044', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00043', timezone('utc'::text, now() - INTERVAL '3 days'), 'Fatigue', 'Moderate', 'Afternoon', 'Mild drowsiness around 3 PM.'),
('d5b38d38-2c26-4d2b-91c6-2c1b97b00045', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00043', timezone('utc'::text, now() - INTERVAL '1 day'), 'Speech', 'Mild', 'Evening', 'Brief word-finding pause during dinner conversation; resolved quickly.')
ON CONFLICT DO NOTHING;

INSERT INTO public.twin_timeline (
    timeline_id, patient_id, evaluation_date, progression_risk_score, triage_category, rano_category, pseudoprogression_risk, is_pseudoprogression, adherence_rate_pct, latent_twin_vector, alert_banner_text
) VALUES (
    'e2b38d38-2c26-4d2b-91c6-2c1b97b00052',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00043',
    '2026-09-03',
    0.4120,
    'Moderate',
    'SD',
    0.2800,
    false,
    100.0,
    '{"latent_dim": 64, "modality_weights": {"mri": 0.50, "nlp": 0.30, "clinical": 0.20}}'::jsonb,
    'Treatment Adherence High: Stable post-surgical cavity volume. Monitor for any motor weakness or speech difficulty.'
) ON CONFLICT DO NOTHING;

-- PATIENT 3: Priya Sharma (MRN: 0044)
INSERT INTO public.patients (
    patient_id, mrn, full_name, age, gender, diagnosis_type, diagnosis_location,
    diagnosis_date, treatment_stage, baseline_kps, oncologist_name, nurse_name, share_research_consent
) VALUES (
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00044',
    '0044',
    'Priya Sharma',
    41,
    'Female',
    'Oligodendroglioma (IDH-Mutant, 1p/19q-codeleted WHO Grade 3)',
    'Right Parietal Lobe',
    '2025-11-20',
    'Surveillance Post-PCV Chemotherapy',
    95,
    'Dr. Aris Thorne, MD (Neuro-Oncology)',
    'Sarah Jensen, RN (Care Navigator)',
    true
) ON CONFLICT (mrn) DO UPDATE SET full_name = EXCLUDED.full_name;

INSERT INTO public.molecular_profiles (
    profile_id, patient_id, idh_status, mgmt_methylated, codeletion_1p19q, assayed_at
) VALUES (
    'f1b38d38-2c26-4d2b-91c6-2c1b97b00003',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00044',
    'Mutant', true, true, '2025-12-01'
) ON CONFLICT DO NOTHING;

INSERT INTO public.mri_scans (
    scan_id, patient_id, scan_date, wt_vol_cm3, tc_vol_cm3, et_vol_cm3, edema_vol_cm3, dice_score, estimated_rcbv, mask_storage_path
) VALUES 
('a6b38d38-2c26-4d2b-91c6-2c1b97b00016', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00044', '2026-03-12', 11.20, 5.40, 1.80, 5.80, 0.9420, 1.25, 'scans/0044/20260312_seg.nii.gz'),
('a7b38d38-2c26-4d2b-91c6-2c1b97b00017', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00044', '2026-08-25', 7.80, 3.10, 0.90, 4.70, 0.9580, 1.10, 'scans/0044/20260825_seg.nii.gz')
ON CONFLICT DO NOTHING;

INSERT INTO public.clinical_reports (
    report_id, patient_id, scan_id, uploaded_by, report_date, raw_text, parsed_entities, plain_language_summary, severity_tag
) VALUES (
    'b3b38d38-2c26-4d2b-91c6-2c1b97b00023',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00044',
    'a7b38d38-2c26-4d2b-91c6-2c1b97b00017',
    'Clinician',
    '2026-08-26',
    'FINDINGS: Continued interval involution of right parietal non-enhancing lesion following completion of PCV protocol. Minimal residual FLAIR hyperintensity measuring 4.7 cm3. No pathological contrast enhancement. Perfusion rCBV normalized at 1.10. No mass effect. IMPRESSION: Favorable durable treatment response; no evidence of recurrent tumor.',
    '{"tumor_location": "Right parietal lobe", "enhancing_dimensions_cm": "None (non-enhancing)", "edema_status": "Minimal residual FLAIR (4.70 cm3)", "mass_effect": false, "midline_shift_mm": 0.0, "rcbv": 1.10, "impression": "Durable complete response"}'::jsonb,
    'Outstanding scan results! Your tumor continues to shrink with no signs of active cancer. The remaining subtle scar tissue is stable and calm, with healthy normal blood flow. Continue your current routine activities with your next check-up in 4 months.',
    'Mild'
) ON CONFLICT DO NOTHING;

INSERT INTO public.medications (
    med_id, patient_id, med_name, dosage, frequency, instructions, prescribed_by, status, adherence_notes, last_updated_by
) VALUES 
('c7b38d38-2c26-4d2b-91c6-2c1b97b00037', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00044', 'Levetiracetam (Keppra)', '500 mg Oral Tablet', 'Twice Daily', 'Maintenance seizure prophylaxis', 'Dr. Aris Thorne, MD', 'Taken', 'Full adherence; seizure-free', 'Doctor')
ON CONFLICT DO NOTHING;

INSERT INTO public.symptom_logs (
    log_id, patient_id, logged_at, symptom_type, severity, onset_time, notes
) VALUES 
('d6b38d38-2c26-4d2b-91c6-2c1b97b00046', 'e5b38d38-2c26-4d2b-91c6-2c1b97b00044', timezone('utc'::text, now() - INTERVAL '5 days'), 'Fatigue', 'Mild', 'Evening', 'Normal tiredness after yoga session; felt fine next day.')
ON CONFLICT DO NOTHING;

INSERT INTO public.twin_timeline (
    timeline_id, patient_id, evaluation_date, progression_risk_score, triage_category, rano_category, pseudoprogression_risk, is_pseudoprogression, adherence_rate_pct, latent_twin_vector, alert_banner_text
) VALUES (
    'e3b38d38-2c26-4d2b-91c6-2c1b97b00053',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00044',
    '2026-08-26',
    0.1150,
    'Mild',
    'CR',
    0.0500,
    false,
    100.0,
    '{"latent_dim": 64, "modality_weights": {"mri": 0.40, "nlp": 0.35, "clinical": 0.25}}'::jsonb,
    'Excellent Stability: Complete treatment response maintained with 100% medication adherence.'
) ON CONFLICT DO NOTHING;
