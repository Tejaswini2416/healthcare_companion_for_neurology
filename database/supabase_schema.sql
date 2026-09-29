-- ============================================================================
-- Healthcare Companion: AI-Powered Cancer Care Companion & Digital Twin for Brain Cancer
-- PostgreSQL / Supabase Schema Definition & Row Level Security (RLS) Policies
-- File: database/supabase_schema.sql
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 1. TABLE: patients
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.patients (
    patient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mrn TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    age INT NOT NULL CHECK (age > 0),
    gender TEXT NOT NULL,
    diagnosis_type TEXT NOT NULL,
    diagnosis_location TEXT NOT NULL,
    diagnosis_date DATE NOT NULL,
    treatment_stage TEXT NOT NULL,
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
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    idh_status TEXT NOT NULL CHECK (idh_status IN ('Mutant', 'Wild-Type')),
    mgmt_methylated BOOLEAN NOT NULL,
    codeletion_1p19q BOOLEAN NOT NULL DEFAULT false,
    assayed_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_molecular_patient ON public.molecular_profiles(patient_id);

-- ============================================================================
-- 3. TABLE: mri_scans
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.mri_scans (
    scan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    scan_date DATE NOT NULL,
    wt_vol_cm3 NUMERIC(8,2) NOT NULL,
    tc_vol_cm3 NUMERIC(8,2) NOT NULL,
    et_vol_cm3 NUMERIC(8,2) NOT NULL,
    edema_vol_cm3 NUMERIC(8,2) NOT NULL,
    dice_score NUMERIC(5,4) NOT NULL DEFAULT 0.8800,
    estimated_rcbv NUMERIC(5,2) NOT NULL DEFAULT 1.50,
    mask_storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_mri_patient_date ON public.mri_scans(patient_id, scan_date);

-- ============================================================================
-- 4. TABLE: clinical_reports
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.clinical_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    scan_id UUID REFERENCES public.mri_scans(scan_id) ON DELETE SET NULL,
    uploaded_by TEXT NOT NULL CHECK (uploaded_by IN ('Clinician', 'Patient')),
    report_date DATE NOT NULL,
    raw_text TEXT NOT NULL,
    parsed_entities JSONB NOT NULL DEFAULT '{}'::jsonb,
    plain_language_summary TEXT NOT NULL,
    severity_tag TEXT NOT NULL CHECK (severity_tag IN ('Mild', 'Moderate', 'Critical')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_reports_patient ON public.clinical_reports(patient_id);

-- ============================================================================
-- 5. TABLE: medications
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.medications (
    med_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    med_name TEXT NOT NULL,
    dosage TEXT NOT NULL,
    frequency TEXT NOT NULL,
    instructions TEXT NOT NULL,
    prescribed_by TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('Taken', 'Due', 'Missed')),
    adherence_notes TEXT,
    last_updated_by TEXT NOT NULL CHECK (last_updated_by IN ('Doctor', 'Patient')),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_meds_patient ON public.medications(patient_id);

-- ============================================================================
-- 6. TABLE: symptom_logs
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.symptom_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    logged_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    symptom_type TEXT NOT NULL CHECK (symptom_type IN ('Headache', 'Vision', 'Seizure', 'Fatigue', 'Speech', 'Motor')),
    severity TEXT NOT NULL CHECK (severity IN ('Mild', 'Moderate', 'Severe')),
    onset_time TIMESTAMPTZ,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_symptoms_patient ON public.symptom_logs(patient_id, logged_at DESC);

-- ============================================================================
-- 7. TABLE: twin_timeline
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.twin_timeline (
    timeline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    evaluation_date DATE NOT NULL,
    progression_risk_score NUMERIC(5,4) NOT NULL,
    triage_category TEXT NOT NULL CHECK (triage_category IN ('Mild', 'Moderate', 'Critical')),
    rano_category TEXT NOT NULL CHECK (rano_category IN ('CR', 'PR', 'SD', 'PD', 'PsP')),
    pseudoprogression_risk NUMERIC(5,4) NOT NULL,
    is_pseudoprogression BOOLEAN NOT NULL DEFAULT false,
    adherence_rate_pct NUMERIC(5,2) NOT NULL DEFAULT 100.00,
    latent_twin_vector JSONB NOT NULL DEFAULT '[]'::jsonb,
    alert_banner_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_twin_patient ON public.twin_timeline(patient_id, evaluation_date DESC);

-- ============================================================================
-- 8. TABLE: counterfactual_simulations
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.counterfactual_simulations (
    sim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    simulation_name TEXT NOT NULL,
    intervention_type TEXT NOT NULL CHECK (intervention_type IN ('SOC_TMZ', 'DOSE_DENSE', 'HOLD_TMZ', 'LOMUSTINE')),
    forecast_horizon_days INT NOT NULL DEFAULT 180,
    predicted_trajectory JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_sims_patient ON public.counterfactual_simulations(patient_id);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.molecular_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mri_scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.clinical_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.medications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.symptom_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.twin_timeline ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.counterfactual_simulations ENABLE ROW LEVEL SECURITY;

-- Allow authenticated read/write for all twin tables
CREATE POLICY "Allow public read access" ON public.patients FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.patients FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access" ON public.patients FOR UPDATE USING (true);

CREATE POLICY "Allow public read access" ON public.molecular_profiles FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.molecular_profiles FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read access" ON public.mri_scans FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.mri_scans FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read access" ON public.clinical_reports FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.clinical_reports FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read access" ON public.medications FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.medications FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access" ON public.medications FOR UPDATE USING (true);

CREATE POLICY "Allow public read access" ON public.symptom_logs FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.symptom_logs FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read access" ON public.twin_timeline FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.twin_timeline FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read access" ON public.counterfactual_simulations FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.counterfactual_simulations FOR INSERT WITH CHECK (true);

-- ============================================================================
-- PRE-SEEDED CLINICAL COHORT PROFILES
-- ============================================================================

-- Patient 0042: V. Thanuja
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
) ON CONFLICT (mrn) DO NOTHING;

INSERT INTO public.molecular_profiles (
    profile_id, patient_id, idh_status, mgmt_methylated, codeletion_1p19q
) VALUES (
    'a1b38d38-2c26-4d2b-91c6-2c1b97b00001',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00042',
    'Mutant',
    true,
    false
) ON CONFLICT DO NOTHING;

-- Patient 0043: Marcus Chen
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
) ON CONFLICT (mrn) DO NOTHING;

INSERT INTO public.molecular_profiles (
    profile_id, patient_id, idh_status, mgmt_methylated, codeletion_1p19q
) VALUES (
    'a2b38d38-2c26-4d2b-91c6-2c1b97b00002',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00043',
    'Wild-Type',
    false,
    false
) ON CONFLICT DO NOTHING;

-- Patient 0044: Priya Sharma
INSERT INTO public.patients (
    patient_id, mrn, full_name, age, gender, diagnosis_type, diagnosis_location,
    diagnosis_date, treatment_stage, baseline_kps, oncologist_name, nurse_name, share_research_consent
) VALUES (
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00044',
    '0044',
    'Priya Sharma',
    41,
    'Female',
    'Oligodendroglioma (WHO Grade 2, IDH-Mutant, 1p/19q-codeleted)',
    'Right Parietal Lobe',
    '2025-11-10',
    'Active Surveillance Post-PCV Chemotherapy',
    90,
    'Dr. Aris Thorne, MD (Neuro-Oncology)',
    'Sarah Jensen, RN (Care Navigator)',
    true
) ON CONFLICT (mrn) DO NOTHING;

INSERT INTO public.molecular_profiles (
    profile_id, patient_id, idh_status, mgmt_methylated, codeletion_1p19q
) VALUES (
    'a3b38d38-2c26-4d2b-91c6-2c1b97b00003',
    'e5b38d38-2c26-4d2b-91c6-2c1b97b00044',
    'Mutant',
    true,
    true
) ON CONFLICT DO NOTHING;
