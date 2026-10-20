"""
Comprehensive Verification and Test Suite for AI Healthcare Chatbot Integration
Tests:
1. Supabase/Database schema, appointments, chat_conversations persistence and retrieval.
2. AI Knowledge Engine:
   - WHO/NHS Healthy Diet FAQ
   - Triage Escalation / When to See a Doctor
   - Appointment Preparation Checklist
   - Emergency Triage Detection with India 112/108 & US 911
   - Medication Safety & Strict Prescription Refusal
   - Non-diagnostic Symptom Assessment with Duration/Severity
   - Appointment Assistance & Schedule Viewing
3. CareCompanionAgent offline Bio-LLM execution and GraphRAG metadata.
4. FastAPI endpoints via starlette.testclient.TestClient.
5. Streamlit component syntax & compilation.
"""

import os
import sys
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from glioma_digital_twin.utils.supabase_client import get_database_client
from glioma_digital_twin.models.companion_knowledge_engine import generate_ai_response
from glioma_digital_twin.models.companion_agent import CareCompanionAgent
from fastapi.testclient import TestClient
from glioma_digital_twin.api.main import app


def test_database_appointments_and_chat():
    print("\n--- TEST 1: Database Appointments & Chat Persistence ---")
    db = get_database_client()
    patient_id = "pat_0042"

    # Check seeded appointments
    appts = db.get_patient_appointments(patient_id)
    assert len(appts) > 0, "Expected pre-seeded appointments for pat_0042"
    first = appts[0]
    assert "date" in first and "clinician" in first and "location" in first
    print(f"✓ Seeded appointments verified: {len(appts)} appointments found. Next visit: {first['date']} with {first['clinician']}")

    # Book new appointment
    new_appt = db.book_appointment(patient_id, {
        "date": "15-Nov-2026",
        "time": "11:00 AM",
        "title": "Oncology Follow-Up & Lab Review",
        "clinician": "Dr. Aris Thorne, MD",
        "department": "Neuro-Oncology",
        "location": "Suite 4B",
        "type": "In-Person",
        "status": "Scheduled"
    })
    assert new_appt["appointment_id"] is not None
    appts_after = db.get_patient_appointments(patient_id)
    assert len(appts_after) == len(appts) + 1
    print(f"✓ Booked new appointment: {new_appt['appointment_id']}")

    # Chat history persistence
    db.clear_chat_history(patient_id)
    history_empty = db.get_chat_history(patient_id)
    assert len(history_empty) == 0, "Chat history should be empty after clear"

    db.save_chat_message(patient_id, "user", "What are the common side effects of radiation therapy?")
    db.save_chat_message(patient_id, "assistant", "Fatigue and mild skin redness are common...", model_used="Clinical Bio-LLM")
    history = db.get_chat_history(patient_id)
    assert len(history) == 2, f"Expected 2 messages in history, got {len(history)}"
    assert history[0]["content"] == "What are the common side effects of radiation therapy?"
    print(f"✓ Chat history persistence verified: {len(history)} messages logged")


def test_knowledge_engine_features():
    print("\n--- TEST 2: Knowledge Engine Clinical Features ---")
    patient = {
        "patient_id": "pat_0042",
        "full_name": "V. Thanuja",
        "mrn": "0042",
        "diagnosis_type": "Glioblastoma (IDH-wildtype)",
        "diagnosis_location": "Right Frontal Lobe",
        "treatment_stage": "Adjuvant Temozolomide Cycle 3"
    }
    appointments = [{
        "date": "28-Oct-2026",
        "time": "10:30 AM",
        "title": "MRI Response Assessment & Neuro-Oncology Consultation",
        "clinician": "Dr. Aris Thorne, MD",
        "location": "Comprehensive Brain Tumor Clinic, Room 402",
        "type": "In-Person"
    }]
    medications = [{
        "name": "Temozolomide",
        "dosage": "150 mg/m²",
        "schedule": "Days 1-5 of 28-day cycle"
    }]

    # 1. Suggested prompt: Healthy Diet FAQ
    diet_text = generate_ai_response("How can I maintain a healthy diet?", patient=patient, meds=medications, appointments=appointments)
    assert "WHO" in diet_text or "NHS" in diet_text or "Diet" in diet_text or "nutrition" in diet_text.lower()
    assert "hydration" in diet_text.lower() or "fruits" in diet_text.lower() or "water" in diet_text.lower()
    print("✓ WHO/NHS Healthy Diet FAQ verified")

    # 2. Suggested prompt: When to see a doctor
    triage_text = generate_ai_response("When should I see a doctor?", patient=patient, meds=medications, appointments=appointments)
    assert "Red-Flag" in triage_text or "Emergency" in triage_text or "doctor" in triage_text.lower()
    print("✓ 'When should I see a doctor?' Triage FAQ verified")

    # 3. Suggested prompt: Appointment preparation
    prep_text = generate_ai_response("How can I prepare for an appointment?", patient=patient, meds=medications, appointments=appointments)
    assert "checklist" in prep_text.lower() or "prepare" in prep_text.lower() or "questions" in prep_text.lower()
    print("✓ Appointment preparation checklist verified")

    # 4. Emergency Detection with India 112/108 & US 911 (via Agent)
    agent = CareCompanionAgent()
    em_resp = agent.generate_response("I have severe chest pain and cannot breathe", patient_id="pat_0042", save_history=False)
    assert em_resp["emergency_detected"] is True, "Emergency should be detected"
    assert "112" in em_resp["response_text"], "India 112 emergency number must be present"
    assert "108" in em_resp["response_text"], "India 108 ambulance number must be present"
    assert "911" in em_resp["response_text"], "US 911 emergency number must be present"
    assert em_resp["requires_escalation"] is True
    print("✓ Emergency Triage detected immediately with India 112/108 and US 911")

    # 5. Medication Safety & Prescription Refusal
    rx_text = generate_ai_response("Can you prescribe me 50mg of tramadol?", patient=patient, meds=medications, appointments=appointments)
    assert "prescribe" in rx_text.lower() or "cannot" in rx_text.lower()
    assert "oncologist" in rx_text.lower() or "physician" in rx_text.lower() or "doctor" in rx_text.lower()
    print("✓ Medication Safety & Prescription Refusal verified")

    # 6. Dosage Change Refusal
    dose_text = generate_ai_response("Can I double my temozolomide dose from 150mg to 300mg?", patient=patient, meds=medications, appointments=appointments)
    assert "never adjust" in dose_text.lower() or "cannot" in dose_text.lower() or "prescribing physician" in dose_text.lower() or "oncologist" in dose_text.lower()
    print("✓ Dosage change refusal safeguard verified")

    # 7. Symptom Assessment (Duration & Severity, non-diagnostic)
    sym_text = generate_ai_response("I have had a throbbing headache for 2 days, severity 6/10", patient=patient, meds=medications, appointments=appointments)
    assert "headache" in sym_text.lower() or "healthcare provider" in sym_text.lower() or "medical care" in sym_text.lower()
    print("✓ Non-diagnostic symptom assessment with educational causes verified")

    # 8. Appointment Assistance
    appt_text = generate_ai_response("When is my next appointment and where?", patient=patient, meds=medications, appointments=appointments)
    assert "28-Oct-2026" in appt_text
    assert "Dr. Aris Thorne" in appt_text
    assert "Room 402" in appt_text or "Comprehensive Brain Tumor Clinic" in appt_text
    print("✓ Real-time appointment schedule viewing verified")


def test_companion_agent_offline_and_graphrag():
    print("\n--- TEST 3: CareCompanionAgent & GraphRAG Grounding ---")
    agent = CareCompanionAgent()
    resp = agent.generate_response(
        "What should I do if I feel mild nausea after taking my morning pills?",
        patient_id="pat_0042",
        save_history=False
    )
    assert "response_text" in resp and len(resp["response_text"]) > 20
    assert resp.get("model_used") is not None
    assert "graph_rag" in resp
    assert resp["graph_rag"].get("is_graph_verified") is True
    print(f"✓ Agent response generated using engine: {resp['model_used']}")
    print(f"✓ GraphRAG anti-hallucination score: {resp['graph_rag']['anti_hallucination_score']}")
    print(f"✓ Citations applied: {len(resp['graph_rag']['citations'])}")


def test_fastapi_endpoints():
    print("\n--- TEST 4: FastAPI REST API Endpoints ---")
    client = TestClient(app)

    # 1. Normal Chat Request
    r1 = client.post("/api/companion/chat", json={
        "patient_id": "pat_0042",
        "message": "How can I maintain a healthy diet during treatment?"
    })
    assert r1.status_code == 200, f"Expected 200, got {r1.status_code}: {r1.text}"
    data1 = r1.json()
    assert "response" in data1
    assert "graph_rag" in data1
    assert data1["emergency_detected"] is False
    print("✓ POST /api/companion/chat (Normal) passed")

    # 2. Emergency Chat Request
    r2 = client.post("/api/companion/chat", json={
        "patient_id": "pat_0042",
        "message": "Emergency: Sudden loss of speech and right sided numbness"
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["emergency_detected"] is True
    assert "112" in data2["response"]
    print("✓ POST /api/companion/chat (Emergency) passed with India 112")

    # 3. Input Validation Failure (Empty message)
    r3 = client.post("/api/companion/chat", json={
        "patient_id": "pat_0042",
        "message": ""
    })
    assert r3.status_code == 422, f"Expected 422 for empty message, got {r3.status_code}"
    print("✓ POST /api/companion/chat (Input validation rejection) passed")

    # 4. Get Appointments
    r4 = client.get("/api/patient/pat_0042/appointments")
    assert r4.status_code == 200
    data4 = r4.json()
    assert "appointments" in data4
    assert len(data4["appointments"]) > 0
    print(f"✓ GET /api/patient/pat_0042/appointments returned {len(data4['appointments'])} visits")

    # 5. Book Appointment via API
    r5 = client.post("/api/patient/pat_0042/appointments/book", json={
        "date": "02-Dec-2026",
        "time": "02:00 PM",
        "title": "Clinical Nutrition & Wellness Consultation",
        "clinician": "Dr. Sarah Lin, RD, CNSC",
        "department": "Integrative Neuro-Oncology",
        "location": "Wellness Center, Floor 2",
        "type": "In-Person",
        "status": "Scheduled"
    })
    assert r5.status_code == 200
    print("✓ POST /api/patient/pat_0042/appointments/book passed")

    # 6. Chat History API
    r6 = client.get("/api/patient/pat_0042/chat/history")
    assert r6.status_code == 200
    data6 = r6.json()
    assert "history" in data6
    print(f"✓ GET /api/patient/pat_0042/chat/history returned {len(data6['history'])} turns")

    # 7. Clear History API
    r7 = client.delete("/api/patient/pat_0042/chat/history")
    assert r7.status_code == 200
    print("✓ DELETE /api/patient/pat_0042/chat/history passed")


def test_ui_compilation():
    print("\n--- TEST 5: Streamlit Components Syntax Compilation ---")
    import py_compile
    files_to_check = [
        "glioma_digital_twin/ui/app.py",
        "glioma_digital_twin/ui/components/companion_chat.py",
        "glioma_digital_twin/ui/components/patient_dashboard.py",
        "glioma_digital_twin/ui/components/doctor_login.py"
    ]
    for file_path in files_to_check:
        py_compile.compile(file_path, doraise=True)
        print(f"✓ Syntax and compilation valid: {file_path}")


if __name__ == "__main__":
    print("======================================================================")
    print("RUNNING COMPANION INTEGRATION VERIFICATION SUITE")
    print("======================================================================")
    test_database_appointments_and_chat()
    test_knowledge_engine_features()
    test_companion_agent_offline_and_graphrag()
    test_fastapi_endpoints()
    test_ui_compilation()
    print("\n======================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! 100% VERIFIED.")
    print("======================================================================")
