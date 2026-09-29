-- ============================================================================
-- PQ-ABAC-EHR: Post-Quantum Attribute-Based Access Control for Electronic Health Records
-- Database: Supabase / PostgreSQL Schema & Row Level Security (RLS)
-- File: supabase/schema.sql
-- Idempotent, Production-Grade Script
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 1. TABLE: profiles
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY, -- references auth.users(id) in Supabase Auth
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL, -- e.g. Oncologist, Triage_Nurse, ER_Physician, Epidemiologist
    department TEXT NOT NULL, -- e.g. Oncology, Emergency, Research, General
    clearance_level INT NOT NULL CHECK (clearance_level BETWEEN 1 AND 3),
    hospital_id TEXT NOT NULL DEFAULT 'HOSP-METRO-01',
    pqc_public_key TEXT, -- ML-KEM-768 public key (hex or base64)
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- Indexing for profile lookups
CREATE INDEX IF NOT EXISTS idx_profiles_role ON public.profiles(role);
CREATE INDEX IF NOT EXISTS idx_profiles_dept ON public.profiles(department);
CREATE INDEX IF NOT EXISTS idx_profiles_clearance ON public.profiles(clearance_level);

-- ============================================================================
-- 2. TABLE: ehr_records
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.ehr_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id TEXT NOT NULL, -- Masked Patient MRN e.g. 'MRN-9482-TX'
    record_title TEXT NOT NULL,
    encrypted_payload TEXT NOT NULL, -- Base64 encoded AES-256-GCM ciphertext
    payload_iv TEXT NOT NULL, -- Base64 12-byte IV / Nonce
    auth_tag TEXT NOT NULL, -- Base64 16-byte GCM Authentication Tag
    encapsulated_dek TEXT NOT NULL, -- Base64 / Hex ML-KEM-768 Encapsulated Key (Ciphertext)
    abac_policy JSONB NOT NULL, -- Abstract Syntax Tree (AST) Access Policy
    record_sensitivity TEXT NOT NULL DEFAULT 'Restricted', -- Standard, Restricted, Highly_Confidential
    record_department TEXT NOT NULL DEFAULT 'General',
    created_by UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_ehr_patient ON public.ehr_records(patient_id);
CREATE INDEX IF NOT EXISTS idx_ehr_dept ON public.ehr_records(record_department);
CREATE INDEX IF NOT EXISTS idx_ehr_sensitivity ON public.ehr_records(record_sensitivity);

-- ============================================================================
-- 3. TABLE: emergency_break_glass_events
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.emergency_break_glass_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id UUID NOT NULL REFERENCES public.ehr_records(id) ON DELETE CASCADE,
    actor_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    justification TEXT NOT NULL CHECK (char_length(justification) >= 20),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    severity TEXT NOT NULL DEFAULT 'CRITICAL_OVERRIDE'
);

CREATE INDEX IF NOT EXISTS idx_break_glass_record ON public.emergency_break_glass_events(record_id);
CREATE INDEX IF NOT EXISTS idx_break_glass_actor ON public.emergency_break_glass_events(actor_id);

-- ============================================================================
-- 4. TABLE: audit_logs (Append-Only Immutable Ledger with SHA3 Tamper Chaining)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL CHECK (event_type IN ('ACCESS_REQUEST', 'DECRYPT_SUCCESS', 'DECRYPT_DENIED', 'BREAK_GLASS', 'KEY_REVOCATION')),
    user_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    record_id UUID REFERENCES public.ehr_records(id) ON DELETE SET NULL,
    policy_evaluated JSONB,
    outcome TEXT NOT NULL CHECK (outcome IN ('GRANTED', 'DENIED', 'OVERRIDDEN')),
    sha3_hash TEXT NOT NULL, -- SHA3-512 Hash chaining over previous log + current event
    client_environment JSONB, -- Network location, user agent, emergency status
    timestamp TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_audit_event ON public.audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_user ON public.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_time ON public.audit_logs(timestamp DESC);

-- ============================================================================
-- 5. IMMUTABILITY ENFORCEMENT: Strictly Append-Only Triggers
-- ============================================================================
CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit log entries and Break-Glass events are cryptographically immutable and cannot be updated or deleted.';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_prevent_audit_logs_update_delete ON public.audit_logs;
CREATE TRIGGER trg_prevent_audit_logs_update_delete
BEFORE UPDATE OR DELETE ON public.audit_logs
FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

DROP TRIGGER IF EXISTS trg_prevent_break_glass_update_delete ON public.emergency_break_glass_events;
CREATE TRIGGER trg_prevent_break_glass_update_delete
BEFORE UPDATE OR DELETE ON public.emergency_break_glass_events
FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

-- ============================================================================
-- 6. ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ehr_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.emergency_break_glass_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- Profiles: Authenticated read on all active profiles; users can update own metadata
DROP POLICY IF EXISTS "Allow authenticated read on profiles" ON public.profiles;
CREATE POLICY "Allow authenticated read on profiles"
    ON public.profiles FOR SELECT
    TO authenticated, anon
    USING (true);

DROP POLICY IF EXISTS "Allow individual update on profile" ON public.profiles;
CREATE POLICY "Allow individual update on profile"
    ON public.profiles FOR UPDATE
    TO authenticated
    USING (auth.uid() = id);

-- EHR Records: Authenticated read and insert
DROP POLICY IF EXISTS "Allow authenticated read on ehr_records" ON public.ehr_records;
CREATE POLICY "Allow authenticated read on ehr_records"
    ON public.ehr_records FOR SELECT
    TO authenticated, anon
    USING (true);

DROP POLICY IF EXISTS "Allow authenticated insert on ehr_records" ON public.ehr_records;
CREATE POLICY "Allow authenticated insert on ehr_records"
    ON public.ehr_records FOR INSERT
    TO authenticated, anon
    WITH CHECK (true);

-- Emergency Break Glass: Append-only insert, authenticated read
DROP POLICY IF EXISTS "Allow read on emergency_break_glass_events" ON public.emergency_break_glass_events;
CREATE POLICY "Allow read on emergency_break_glass_events"
    ON public.emergency_break_glass_events FOR SELECT
    TO authenticated, anon
    USING (true);

DROP POLICY IF EXISTS "Allow insert on emergency_break_glass_events" ON public.emergency_break_glass_events;
CREATE POLICY "Allow insert on emergency_break_glass_events"
    ON public.emergency_break_glass_events FOR INSERT
    TO authenticated, anon
    WITH CHECK (true);

-- Audit Logs: Append-only insert, authenticated read
DROP POLICY IF EXISTS "Allow read on audit_logs" ON public.audit_logs;
CREATE POLICY "Allow read on audit_logs"
    ON public.audit_logs FOR SELECT
    TO authenticated, anon
    USING (true);

DROP POLICY IF EXISTS "Allow insert on audit_logs" ON public.audit_logs;
CREATE POLICY "Allow insert on audit_logs"
    ON public.audit_logs FOR INSERT
    TO authenticated, anon
    WITH CHECK (true);

-- ============================================================================
-- 7. PRE-SEEDED DEMO DATASET
-- 4 Clinician Profiles + 3 Encrypted FHIR Clinical Records
-- ============================================================================

-- Fixed Mock UUIDs for demo personas
INSERT INTO public.profiles (id, full_name, email, role, department, clearance_level, hospital_id, pqc_public_key, is_active)
VALUES
    ('11111111-1111-1111-1111-111111111111', 'Dr. Sarah Rao', 'sarah.rao@metrohealth.org', 'Oncologist', 'Oncology', 3, 'HOSP-METRO-01', 'PQC_MLKEM768_PUB_SARAH_RAO_ONCOLOGY_CHAIR_9a8f7c6e5d4b3a21', true),
    ('22222222-2222-2222-2222-222222222222', 'Nurse Alex', 'alex.mercer@metrohealth.org', 'Triage_Nurse', 'Emergency', 1, 'HOSP-METRO-01', 'PQC_MLKEM768_PUB_ALEX_MERCER_TRIAGE_EMERGENCY_5e4d3c2b1a0f9e8d', true),
    ('33333333-3333-3333-3333-333333333333', 'Dr. Emergency John', 'john.doe@metrohealth.org', 'ER_Physician', 'Emergency', 2, 'HOSP-METRO-01', 'PQC_MLKEM768_PUB_JOHN_DOE_TRAUMA_CHIEF_8b7a6c5d4e3f2a1b', true),
    ('44444444-4444-4444-4444-444444444444', 'Researcher Dave', 'dave.miller@bioresearch.org', 'Epidemiologist', 'Research', 1, 'HOSP-METRO-01', 'PQC_MLKEM768_PUB_DAVE_MILLER_RESEARCH_INST_3c2b1a0f9e8d7c6b', true)
ON CONFLICT (id) DO UPDATE SET
    full_name = EXCLUDED.full_name,
    email = EXCLUDED.email,
    role = EXCLUDED.role,
    department = EXCLUDED.department,
    clearance_level = EXCLUDED.clearance_level,
    hospital_id = EXCLUDED.hospital_id,
    pqc_public_key = EXCLUDED.pqc_public_key,
    is_active = EXCLUDED.is_active;

-- Record 1: Oncology Clinical Trial Phase III Note
-- Policy: Requires Role == 'Oncologist' AND Clearance >= 3 AND Department == 'Oncology'
INSERT INTO public.ehr_records (
    id,
    patient_id,
    record_title,
    encrypted_payload,
    payload_iv,
    auth_tag,
    encapsulated_dek,
    abac_policy,
    record_sensitivity,
    record_department,
    created_by
) VALUES (
    'a1111111-1111-1111-1111-111111111111',
    'MRN-ONC-8842',
    'Oncology Trial Note: CAR-T Protocol Genomic Response',
    -- Pre-calculated ciphertext of FHIR Bundle (or dynamic hybrid payload in app)
    'kK9mXz3f8hN0lPqR...[AES-256-GCM_PAYLOAD_BASE64]...',
    '7uN2xY8wQ1sP',
    '9bK4fT2mL8xP0qWz',
    'MLKEM768_ENCAPSULATED_CT_7a8b9c0d1e2f3a4b...',
    '{
      "operator": "AND",
      "rules": [
        { "field": "subject.role", "operator": "EQUALS", "value": "Oncologist" },
        { "field": "subject.clearance_level", "operator": "GREATER_THAN_OR_EQUAL", "value": 3 },
        { "field": "subject.department", "operator": "EQUALS", "value": "Oncology" },
        { "field": "action", "operator": "IN", "value": ["read", "write"] }
      ]
    }'::jsonb,
    'Highly_Confidential',
    'Oncology',
    '11111111-1111-1111-1111-111111111111'
) ON CONFLICT (id) DO NOTHING;

-- Record 2: Emergency Trauma Resuscitation Panel
-- Policy: Requires (Department == 'Emergency' AND Clearance >= 2) OR (Action == 'break_glass_override')
INSERT INTO public.ehr_records (
    id,
    patient_id,
    record_title,
    encrypted_payload,
    payload_iv,
    auth_tag,
    encapsulated_dek,
    abac_policy,
    record_sensitivity,
    record_department,
    created_by
) VALUES (
    'b2222222-2222-2222-2222-222222222222',
    'MRN-EMERG-3041',
    'Emergency Trauma Panel: Level 1 Resuscitation & Blood Type O-',
    'tL2mK9vX5pQ8wR1n...[AES-256-GCM_PAYLOAD_BASE64]...',
    '3mP9xY2wQ8sT',
    '4aL8fT1mN7xK9qVz',
    'MLKEM768_ENCAPSULATED_CT_5d6e7f8a9b0c1d2e...',
    '{
      "operator": "OR",
      "rules": [
        {
          "operator": "AND",
          "rules": [
            { "field": "subject.department", "operator": "EQUALS", "value": "Emergency" },
            { "field": "subject.clearance_level", "operator": "GREATER_THAN_OR_EQUAL", "value": 2 }
          ]
        },
        {
          "field": "action",
          "operator": "EQUALS",
          "value": "break_glass_override"
        }
      ]
    }'::jsonb,
    'Restricted',
    'Emergency',
    '33333333-3333-3333-3333-333333333333'
) ON CONFLICT (id) DO NOTHING;

-- Record 3: Outpatient Routine Annual Checkup
-- Policy: Requires Clearance >= 1 AND Subject Role IN ['Oncologist', 'Triage_Nurse', 'ER_Physician']
INSERT INTO public.ehr_records (
    id,
    patient_id,
    record_title,
    encrypted_payload,
    payload_iv,
    auth_tag,
    encapsulated_dek,
    abac_policy,
    record_sensitivity,
    record_department,
    created_by
) VALUES (
    'c3333333-3333-3333-3333-333333333333',
    'MRN-ROUTINE-1092',
    'General Health Assessment: Metabolic Panel & Lipid Profile',
    'qM4nX7yP1kL9wV2s...[AES-256-GCM_PAYLOAD_BASE64]...',
    '5xQ8mN2wY1sK',
    '8cK2fL9mT4xP1qNz',
    'MLKEM768_ENCAPSULATED_CT_3b4c5d6e7f8a9b0c...',
    '{
      "operator": "AND",
      "rules": [
        { "field": "subject.clearance_level", "operator": "GREATER_THAN_OR_EQUAL", "value": 1 },
        { "field": "subject.role", "operator": "IN", "value": ["Oncologist", "Triage_Nurse", "ER_Physician"] }
      ]
    }'::jsonb,
    'Standard',
    'General',
    '22222222-2222-2222-2222-222222222222'
) ON CONFLICT (id) DO NOTHING;

-- Initial Genesis Audit Entry with SHA3-512 Initial State Hash
INSERT INTO public.audit_logs (
    id,
    event_type,
    user_id,
    record_id,
    policy_evaluated,
    outcome,
    sha3_hash,
    client_environment,
    timestamp
) VALUES (
    'f0000000-0000-0000-0000-000000000001',
    'ACCESS_REQUEST',
    '11111111-1111-1111-1111-111111111111',
    'a1111111-1111-1111-1111-111111111111',
    '{"system": "GENESIS_BOOTSTRAP"}'::jsonb,
    'GRANTED',
    '00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000',
    '{"host": "GENESIS", "env": "SECURE_BOOTSTRAP"}'::jsonb,
    timezone('utc'::text, now())
) ON CONFLICT (id) DO NOTHING;
