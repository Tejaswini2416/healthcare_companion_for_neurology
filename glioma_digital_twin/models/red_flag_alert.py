"""
Automated Red-Flag Escalation Protocol Module
Implements hospital-grade automated escalation workflows for acute neuro-oncology complications:
1. Multi-tiered clinical urgency classification (Level 1: Routine, Level 2: Urgent, Level 3: STAT Emergency).
2. SBAR (Situation, Background, Assessment, Recommendation) structured clinical handoff generator.
3. Multi-channel dispatch simulator (Hospital Pager / PagerDuty Webhook + SMS Gateway).
4. Attending physician acknowledgement and active escalation queue management.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class RedFlagEscalationEngine:
    """
    Evaluates acute patient symptoms, MRI findings, and digital twin trajectories
    to trigger automated clinical escalation protocols.
    """
    def __init__(self):
        # In-memory escalation events store (singleton-like state)
        if "global_escalation_queue" not in globals():
            globals()["global_escalation_queue"] = []
        self.queue: List[Dict[str, Any]] = globals()["global_escalation_queue"]

        # Seed with initial demo escalation if empty
        if not self.queue:
            self._seed_initial_demo_escalation()

    def _seed_initial_demo_escalation(self):
        """Pre-seeds an active clinical escalation for demonstration."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.queue.append({
            "alert_id": "esc-0042-01",
            "patient_id": "e5b38d38-2c26-4d2b-91c6-2c1b97b00042",
            "patient_name": "V. Thanuja",
            "mrn": "0042",
            "severity_level": "Level 2 (Urgent Care Team Escalation)",
            "severity_tier": "Urgent",
            "trigger_event": "Sensory Seizure Aura + Missed Levetiracetam (Keppra)",
            "timestamp": now_str,
            "status": "Active (Pending Physician Acknowledgment)",
            "sbar": {
                "situation": "24yo F patient with Left Temporal IDH-mutant glioma reports olfactory aura and right-hand tingling.",
                "background": "Patient missed morning dose of Levetiracetam (750mg) due to nausea. Receiving Stupp Cycle 3 adjuvant TMZ.",
                "assessment": "Missed anticonvulsant precipitously lowering seizure threshold with break-through sensory aura.",
                "recommendation": "Urgent nurse call to administer compensatory dose; instruct patient on aura precautions and rescue protocol."
            },
            "dispatch_channels": {
                "sms": {
                    "to": "+1 (555) 019-8234 (On-Call Neuro-Oncology Nurse Navigator)",
                    "body": "[URGENT CARE ALERT] Patient V. Thanuja (MRN: 0042) logged sensory seizure aura following missed Keppra. Review SBAR in workstation.",
                    "status": "Delivered"
                },
                "pager_webhook": {
                    "endpoint": "https://api.hospital-pagerduty.org/v2/enqueue",
                    "priority": "P2_HIGH",
                    "service": "Neuro-Oncology In-Basket Care Navigation",
                    "status": "HTTP 200 Accepted"
                }
            },
            "acknowledged_by": None,
            "resolution_notes": None
        })

    def evaluate_symptom_event(
        self,
        symptom_type: str,
        severity: str,
        notes: str,
        patient_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates a logged symptom for acute neurological red flags.
        Returns escalation packet if triggered, or None.
        """
        sym_lower = (symptom_type + " " + notes).lower()
        mrn = patient_data.get("mrn", "0042")
        p_name = patient_data.get("full_name", "Patient")
        p_diag = patient_data.get("diagnosis_type", "High-Grade Glioma")
        oncologist = patient_data.get("oncologist_name", "Dr. Aris Thorne, MD")

        # 1. Level 3 (STAT / Immediate ER) Triggers
        is_stat = False
        stat_reason = ""
        if any(w in sym_lower for w in ["cannot speak", "aphasia", "loss of speech", "speech arrest"]):
            is_stat = True
            stat_reason = "Acute Expressive Aphasia / Speech Arrest"
        elif any(w in sym_lower for w in ["cannot move", "paralysis", "hemiparesis", "weak arm", "facial droop"]):
            is_stat = True
            stat_reason = "Acute Focal Motor Deficit / Hemiparesis"
        elif any(w in sym_lower for w in ["repeated seizure", "grand mal", "status epilepticus", "passed out", "unconscious"]):
            is_stat = True
            stat_reason = "Prolonged / Convulsive Seizure Activity"
        elif any(w in sym_lower for w in ["worst headache", "projectile vomit", "severe vomiting and headache"]):
            is_stat = True
            stat_reason = "Acute Intracranial Pressure Decompensation"

        if is_stat:
            alert = self._create_escalation_record(
                patient_data=patient_data,
                severity_level="Level 3 (STAT Emergency)",
                severity_tier="STAT",
                trigger_event=stat_reason,
                situation=f"EMERGENT: {p_name} (MRN: {mrn}) presents with {stat_reason}.",
                background=f"{p_diag}. Under active treatment with {oncologist}.",
                assessment="Impending acute neurological decompensation or herniation. Urgent clinical stabilization required.",
                recommendation="Direct patient to nearest emergency department (call 911). Page on-call neurosurgical resident and attending immediately."
            )
            return alert

        # 2. Level 2 (Urgent Care Team Escalation within 4 hours)
        is_urgent = False
        urgent_reason = ""
        if any(w in sym_lower for w in ["seizure", "aura", "twitching", "smell", "tingling"]):
            is_urgent = True
            urgent_reason = "Sensory Seizure Aura / Breakthrough Focal Episode"
        elif any(w in sym_lower for w in ["vision", "blurred vision", "double vision", "diplopia"]):
            is_urgent = True
            urgent_reason = "New Onset Visual Field Deficit"
        elif severity == "Severe":
            is_urgent = True
            urgent_reason = f"Severe {symptom_type} Reported"

        if is_urgent:
            alert = self._create_escalation_record(
                patient_data=patient_data,
                severity_level="Level 2 (Urgent Care Team Escalation)",
                severity_tier="Urgent",
                trigger_event=urgent_reason,
                situation=f"{p_name} (MRN: {mrn}) reports {urgent_reason}.",
                background=f"Patient diagnosed with {p_diag}. Under supervision of {oncologist}.",
                assessment=f"Patient experiencing significant clinical symptom escalation ({symptom_type}, Severity: {severity}).",
                recommendation="Urgent contact by nurse navigator within 4 hours. Review antiepileptic adherence and consider Dexamethasone adjustment."
            )
            return alert

        return None

    def _create_escalation_record(
        self,
        patient_data: Dict[str, Any],
        severity_level: str,
        severity_tier: str,
        trigger_event: str,
        situation: str,
        background: str,
        assessment: str,
        recommendation: str
    ) -> Dict[str, Any]:
        """Creates and dispatches an escalation record."""
        alert_id = f"esc-{patient_data.get('mrn', '0042')}-{uuid.uuid4().hex[:6]}"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        sms_body = (
            f"[{severity_tier.upper()} NEURO-ONCOLOGY ALERT] "
            f"Patient: {patient_data.get('full_name')} (MRN: {patient_data.get('mrn')}). "
            f"Trigger: {trigger_event}. SBAR generated in clinical workstation."
        )

        escalation_record = {
            "alert_id": alert_id,
            "patient_id": patient_data.get("patient_id"),
            "patient_name": patient_data.get("full_name"),
            "mrn": patient_data.get("mrn"),
            "severity_level": severity_level,
            "severity_tier": severity_tier,
            "trigger_event": trigger_event,
            "timestamp": now_str,
            "status": "Active (Pending Physician Acknowledgment)",
            "sbar": {
                "situation": situation,
                "background": background,
                "assessment": assessment,
                "recommendation": recommendation
            },
            "dispatch_channels": {
                "sms": {
                    "to": "+1 (555) 019-8492 (Attending Neuro-Oncologist On-Call)",
                    "body": sms_body,
                    "status": "Delivered"
                },
                "pager_webhook": {
                    "endpoint": "https://api.hospital-paging.org/v1/dispatch",
                    "priority": "P1_STAT" if severity_tier == "STAT" else "P2_HIGH",
                    "service": "Comprehensive Neuro-Oncology Triage",
                    "status": "HTTP 200 Broadcasted"
                }
            },
            "acknowledged_by": None,
            "resolution_notes": None
        }

        self.queue.insert(0, escalation_record)
        return escalation_record

    def get_all_escalations(self, mrn_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves escalation alerts, optionally filtered by patient MRN."""
        if mrn_filter:
            return [e for e in self.queue if e.get("mrn") == mrn_filter]
        return self.queue

    def acknowledge_alert(self, alert_id: str, doctor_name: str, resolution_notes: str) -> bool:
        """Records attending physician clinical acknowledgement and resolution."""
        for e in self.queue:
            if e["alert_id"] == alert_id:
                e["status"] = f"Resolved (Acknowledged by {doctor_name})"
                e["acknowledged_by"] = doctor_name
                e["acknowledged_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                e["resolution_notes"] = resolution_notes
                return True
        return False


# Singleton engine getter
_global_escalation_engine = None

def get_escalation_engine() -> RedFlagEscalationEngine:
    global _global_escalation_engine
    if _global_escalation_engine is None:
        _global_escalation_engine = RedFlagEscalationEngine()
    return _global_escalation_engine
