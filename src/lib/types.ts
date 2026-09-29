// ============================================================================
// PQ-ABAC-EHR: Core Type Definitions
// ============================================================================

export type ClinicianRole = 
  | 'Oncologist'
  | 'Triage_Nurse'
  | 'ER_Physician'
  | 'Epidemiologist'
  | 'Chief_Medical_Officer'
  | 'Radiologist';

export type Department = 
  | 'Oncology'
  | 'Emergency'
  | 'Research'
  | 'General'
  | 'Cardiology'
  | 'Pediatrics';

export type ClearanceLevel = 1 | 2 | 3;

export type RecordSensitivity = 
  | 'Standard'
  | 'Restricted'
  | 'Highly_Confidential';

export type AccessAction = 
  | 'read'
  | 'write'
  | 'export'
  | 'break_glass_override';

export type NetworkLocation = 'Intranet' | 'Extranet';

export interface SubjectAttributes {
  id: string;
  full_name: string;
  role: ClinicianRole;
  department: Department;
  clearance_level: ClearanceLevel;
  hospital_id: string;
  is_active: boolean;
}

export interface ObjectAttributes {
  record_id: string;
  patient_id: string;
  record_title: string;
  record_sensitivity: RecordSensitivity;
  record_department: Department;
  patient_consent_status: 'Granted' | 'Revoked' | 'Emergency_Only';
}

export interface EnvironmentAttributes {
  current_time: string; // ISO 8601 string
  network_location: NetworkLocation;
  emergency_state_active: boolean;
}

export interface EvaluationContext {
  subject: SubjectAttributes;
  object: ObjectAttributes;
  action: AccessAction;
  environment: EnvironmentAttributes;
}

// ============================================================================
// ABAC Policy AST (Abstract Syntax Tree) Types
// ============================================================================

export type ComparisonOperator = 
  | 'EQUALS'
  | 'NOT_EQUALS'
  | 'IN'
  | 'NOT_IN'
  | 'GREATER_THAN_OR_EQUAL'
  | 'LESS_THAN_OR_EQUAL'
  | 'GREATER_THAN'
  | 'LESS_THAN';

export type LogicalOperator = 'AND' | 'OR' | 'NOT';

export interface PolicyRuleLeaf {
  field: string; // e.g. "subject.role", "subject.clearance_level", "action"
  operator: ComparisonOperator;
  value: string | number | boolean | string[] | number[];
}

export interface PolicyNodeGroup {
  operator: LogicalOperator;
  rules: (PolicyRuleLeaf | PolicyNodeGroup)[];
}

export type PolicyAST = PolicyRuleLeaf | PolicyNodeGroup;

export interface AttributeLineageItem {
  field: string;
  operator: string;
  expectedValue: any;
  actualValue: any;
  satisfied: boolean;
  diagnostic: string;
}

export interface ABACEvaluationResult {
  status: 'PERMIT' | 'DENY';
  reason?: string;
  lineage: AttributeLineageItem[];
  failedPredicates: string[];
  evaluatedAt: string;
}

// ============================================================================
// FHIR Clinical Record Payload
// ============================================================================

export interface FHIRCondition {
  code: string;
  display: string;
  clinicalStatus: 'active' | 'recurrence' | 'relapse' | 'remission' | 'resolved';
  verificationStatus: 'confirmed' | 'provisional' | 'differential';
  onsetDate?: string;
}

export interface FHIRMedication {
  medicationName: string;
  dosage: string;
  frequency: string;
  route: string;
  prescribedDate: string;
}

export interface FHIRVitals {
  heartRate: number; // bpm
  bloodPressure: string; // e.g. "120/80"
  respiratoryRate: number; // breaths/min
  temperature: number; // Celsius
  oxygenSaturation: number; // %
  bloodType: string; // e.g. "O-", "A+"
  resuscitationStatus: 'FULL_CODE' | 'DNR' | 'DNI' | 'COMFORT_MEASURES';
}

export interface ClinicalRecordPayload {
  resourceType: 'Bundle';
  patientId: string;
  patientName: string;
  dateOfBirth: string;
  gender: string;
  conditions: FHIRCondition[];
  medications: FHIRMedication[];
  vitals: FHIRVitals;
  allergies: string[];
  clinicalNotes: string;
  confidentialTrialData?: {
    trialId: string;
    phase: string;
    genomicMarker: string;
    experimentalTherapy: string;
  };
}

// ============================================================================
// Database & Storage Models
// ============================================================================

export interface UserProfile {
  id: string;
  full_name: string;
  email: string;
  role: ClinicianRole;
  department: Department;
  clearance_level: ClearanceLevel;
  hospital_id: string;
  pqc_public_key: string; // Base64 or Hex
  is_active: boolean;
  created_at: string;
}

export interface EHRRecord {
  id: string;
  patient_id: string;
  record_title: string;
  encrypted_payload: string; // Base64 AES-256-GCM ciphertext
  payload_iv: string; // Base64 12-byte IV
  auth_tag: string; // Base64 16-byte GCM tag
  encapsulated_dek: string; // Base64 ML-KEM-768 ciphertext (1088 bytes)
  wrapped_dek?: string; // Base64 encrypted DEK (sealed via HKDF key)
  abac_policy: PolicyAST;
  record_sensitivity: RecordSensitivity;
  record_department: Department;
  created_by?: string;
  created_at: string;
  // Unencrypted reference payload for initial demo seeding
  _plaintext_cache?: ClinicalRecordPayload;
}

export interface EmergencyBreakGlassEvent {
  id: string;
  record_id: string;
  actor_id: string;
  actor_name?: string;
  justification: string;
  timestamp: string;
  severity: 'CRITICAL_OVERRIDE';
}

export interface AuditLogEntry {
  id: string;
  event_type: 'ACCESS_REQUEST' | 'DECRYPT_SUCCESS' | 'DECRYPT_DENIED' | 'BREAK_GLASS' | 'KEY_REVOCATION';
  user_id: string;
  user_name?: string;
  record_id?: string;
  record_title?: string;
  policy_evaluated?: any;
  outcome: 'GRANTED' | 'DENIED' | 'OVERRIDDEN';
  sha3_hash: string;
  previous_hash?: string;
  client_environment?: {
    network_location: string;
    emergency_state: boolean;
    userAgent?: string;
  };
  timestamp: string;
}

// ============================================================================
// Cryptographic Envelope & Benchmark Data
// ============================================================================

export interface EncryptedEnvelope {
  ciphertext: string; // Base64
  iv: string; // Base64
  authTag: string; // Base64
  encapsulatedDek: string; // Base64 ML-KEM-768 ciphertext
  wrappedDek: string; // Base64 sealed DEK
  dekFingerprint: string; // SHA3-512 hex
}

export interface BenchmarkMetrics {
  algorithm: string;
  category: 'Classical' | 'Post-Quantum' | 'Symmetric';
  publicKeySize: number; // bytes
  ciphertextOrSignatureSize: number; // bytes
  operationLatencyMs: number;
  quantumSecurityBits: number;
}
