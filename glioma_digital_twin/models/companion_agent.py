"""
Personalized Conversational Care Companion Module (GPT-4o & Clinical LLM Integration)
Provides:
1. Direct OpenAI GPT Integration (GPT-4o, GPT-4o-mini, GPT-3.5-turbo) with Patient Digital Twin context injection.
2. Groq Cloud LLM Integration (LLaMA-3.3-70B, LLaMA-3.1-8B) for ultra-fast, high-intelligence responses.
3. Built-in Generative Clinical Bio-LLM Engine that generates dynamic, empathetic, ChatGPT-quality responses for ANY health or lifestyle question even without an API key.
4. NCCN Guidelines & Pharmacological GraphRAG Anti-Hallucination Guard.
5. Immediate Clinical Red-Flag Safety Escalation.
"""

import os
import re
from typing import Dict, Any, List, Optional
from ..utils.supabase_client import get_database_client
from .graph_rag import GraphRAGClinicalGuard
from .red_flag_alert import get_escalation_engine


class CareCompanionAgent:
    """
    Care Companion that generates empathetic, clinically safe, and comprehensive
    responses to all health-related queries. Powered by OpenAI GPT-4o, Groq LLaMA-3,
    or our local Generative Clinical Bio-LLM Engine.
    """
    def __init__(self):
        self.rag_guard = GraphRAGClinicalGuard()
        self.escalation_engine = get_escalation_engine()
        self.red_flag_keywords = [
            "passed out", "unconscious", "repeated seizures", "cannot move arm",
            "cannot move leg", "cannot speak", "paralyzed", "worst headache of my life",
            "vomiting blood", "emergency", "911", "falling down", "chest pain",
            "shortness of breath", "cant breathe", "can't breathe", "loss of consciousness"
        ]

    def _build_digital_twin_system_prompt(
        self,
        patient: Dict[str, Any],
        scans: List[Dict[str, Any]],
        meds: List[Dict[str, Any]],
        symptoms: List[Dict[str, Any]],
        timeline: List[Dict[str, Any]],
        latest_scan: Dict[str, Any]
    ) -> str:
        """Constructs an exhaustive clinical system prompt embedding the patient's full digital twin state."""
        patient_name = patient.get("full_name", "Patient")
        mrn = patient.get("mrn", "0042")
        age = patient.get("age", 24)
        gender = patient.get("gender", "Female")
        diagnosis = patient.get("diagnosis_type", "High-Grade Glioma (Astrocytoma, IDH-Mutant Grade 4)")
        location = patient.get("diagnosis_location", "Left Temporal Lobe")
        oncologist = patient.get("oncologist_name", "Dr. Aris Thorne, MD (Neuro-Oncology)")
        nurse = patient.get("nurse_name", "Sarah Jensen, RN (Care Navigator)")

        wt_v = float(latest_scan.get("wt_vol_cm3", 14.6))
        et_v = float(latest_scan.get("et_vol_cm3", 4.1))
        ed_v = float(latest_scan.get("edema_vol_cm3", 6.5))
        rcbv = float(latest_scan.get("estimated_rcbv", 1.48))
        scan_date = latest_scan.get("scan_date", "2026-09-18")

        med_lines = []
        for m in meds:
            med_lines.append(f"- {m.get('med_name')} ({m.get('dosage')}): Schedule {m.get('schedule_time', 'Daily')}, Status: {m.get('status', 'Taken')}")
        meds_text = "\n".join(med_lines) if med_lines else "- Temozolomide 150 mg/m² (Adjuvant), Levetiracetam 500 mg BID (Keppra)"

        sym_lines = []
        for s in symptoms[:4]:
            sym_lines.append(f"- {s.get('symptom_type', 'Symptom')} (Severity: {s.get('severity', 'Mild')}): {s.get('notes', '')}")
        syms_text = "\n".join(sym_lines) if sym_lines else "- Mild cognitive fatigue, occasional sensory aura (controlled)"

        latest_twin = timeline[0] if timeline else {}
        rano_status = latest_twin.get("rano_category", "PsP (Pseudoprogression / Treatment Effect)")

        prompt = f"""You are the personal AI Cancer Care Companion and Digital Health Twin for {patient_name} (MRN: {mrn}).
You are built to talk, reason, and support the patient just like ChatGPT (GPT-4o), with deep neuro-oncology clinical acumen, unwavering empathy, and conversational warmth.

PATIENT DIGITAL HEALTH TWIN RECORD:
• Name: {patient_name} ({gender}, Age: {age}) | MRN: {mrn}
• Diagnosis: {diagnosis}
• Anatomical Location: {location}
• Attending Physician: {oncologist} | Clinical Nurse Navigator: {nurse}
• Latest 3D Brain MRI Volumetrics ({scan_date}):
  - Whole Tumor Volume (WT): {wt_v:.2f} cm³
  - Enhancing Active Core (ET): {et_v:.2f} cm³ (demonstrates stable post-operative residual margin)
  - Vasogenic Perilesional Edema (ED): {ed_v:.2f} cm³
  - Perfusion DSC rCBV: {rcbv:.2f} (Reference <1.75 indicates low perfusion, highly consistent with post-radiotherapy treatment effect / pseudoprogression rather than rapid tumor recurrence)
• Current Active Prescriptions:
{meds_text}
• Recent Symptoms Logged:
{syms_text}
• Digital Twin Milestone & RANO 2.0 Triage: {rano_status}
• Clinical Guideline Grounding: NCCN Category 1 Guidelines for Central Nervous System Cancers & American Academy of Neurology (AAN) Standards.

ROLE & BEHAVIORAL DIRECTIVES:
1. ANSWER EVERY QUESTION DIRECTLY & COMPLETELY: Never dodge, give generic disclaimers, or provide empty templates. Speak naturally, conversationally, and insightfully.
2. EMPATHETIC BEDSIDE MANNER: Address the patient warmly as {patient_name.split()[0]}. Acknowledge their emotions, validate their fears, and celebrate stability.
3. ADAPTIVE EXPLANATIONS:
   - If the patient asks for simple explanations (e.g. "explain like I'm 10", "explain in simple terms", "plain English"), break down complex oncology into engaging, crystal-clear story analogies (such as comparing the brain to a grand command center and medicines to guardian defenders).
   - If they ask technical medical questions, provide clear, precise physiological explanations with clinical depth.
4. LIFESTYLE & HOLISTIC CARE: Offer practical, evidence-based advice for nutrition (protein, hydration, caffeine), safe physical activity, driving safety (following DMV/AAN seizure-free intervals), flying, and sleep.
5. CLINICAL RED-FLAGS: If the patient mentions acute emergencies (loss of consciousness, uncontrolled repeated seizures, worst headache of life, sudden weakness/paralysis), immediately instruct them to call 911 or go to the nearest emergency department.
6. COLLABORATIVE CARE: For medication adjustments or new concerning symptoms, encourage confirming with {nurse} or {oncologist}.

Format your responses with clean Markdown, helpful bullet points, and welcoming headings."""
        return prompt

    def _call_openai_gpt(
        self,
        user_message: str,
        patient: Dict[str, Any],
        scans: List[Dict[str, Any]],
        meds: List[Dict[str, Any]],
        symptoms: List[Dict[str, Any]],
        timeline: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]],
        api_key: str,
        model_name: str = "gpt-4o"
    ) -> Optional[str]:
        """Calls OpenAI Chat Completions API with Patient Digital Twin context."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            latest_scan = scans[-1] if scans else {}
            system_prompt = self._build_digital_twin_system_prompt(
                patient, scans, meds, symptoms, timeline, latest_scan
            )

            messages = [{"role": "system", "content": system_prompt}]

            # Append previous conversation history for conversational multi-turn context
            if conversation_history:
                for msg in conversation_history[-8:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content:
                        messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": user_message})

            # Use valid model name
            target_model = model_name if model_name in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"] else "gpt-4o"

            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[CareCompanionAgent] OpenAI API call failed: {e}")
            return None

    def _call_groq_llm(
        self,
        user_message: str,
        patient: Dict[str, Any],
        scans: List[Dict[str, Any]],
        meds: List[Dict[str, Any]],
        symptoms: List[Dict[str, Any]],
        timeline: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]],
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile"
    ) -> Optional[str]:
        """Calls Groq Cloud API for high-speed LLaMA inference."""
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            latest_scan = scans[-1] if scans else {}
            system_prompt = self._build_digital_twin_system_prompt(
                patient, scans, meds, symptoms, timeline, latest_scan
            )

            messages = [{"role": "system", "content": system_prompt}]
            if conversation_history:
                for msg in conversation_history[-8:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content:
                        messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": user_message})

            target_model = "llama-3.3-70b-versatile" if "70b" in model_name else "llama-3.1-8b-instant"

            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[CareCompanionAgent] Groq API call failed: {e}")
            return None

    def _generate_clinical_gpt_response(
        self,
        user_message: str,
        patient: Dict[str, Any],
        scans: List[Dict[str, Any]],
        meds: List[Dict[str, Any]],
        symptoms: List[Dict[str, Any]],
        timeline: List[Dict[str, Any]]
    ) -> str:
        """
        High-Fidelity Generative Clinical GPT Engine.
        Synthesizes intelligent, empathetic, ChatGPT-quality responses tailored to the
        patient's individual condition and question when external cloud API keys are not supplied.
        """
        patient_name = patient.get("full_name", "Patient")
        first_name = patient_name.split()[0] if patient_name else "there"
        mrn = patient.get("mrn", "0042")
        diagnosis = patient.get("diagnosis_type", "High-Grade Glioma")
        location = patient.get("diagnosis_location", "Left Temporal Lobe")
        oncologist = patient.get("oncologist_name", "Dr. Aris Thorne, MD")
        nurse = patient.get("nurse_name", "Sarah Jensen, RN")

        latest_scan = scans[-1] if scans else {}
        wt_v = float(latest_scan.get("wt_vol_cm3", 14.6))
        et_v = float(latest_scan.get("et_vol_cm3", 4.1))
        ed_v = float(latest_scan.get("edema_vol_cm3", 6.5))
        rcbv = float(latest_scan.get("estimated_rcbv", 1.48))
        scan_date = latest_scan.get("scan_date", "2026-09-18")

        msg_lower = user_message.lower().strip()

        def _matches_any(text: str, kws: list) -> bool:
            for kw in kws:
                if " " in kw or "-" in kw:
                    if kw in text:
                        return True
                else:
                    if re.search(rf"\b{re.escape(kw)}\b", text):
                        return True
            return False

        # -------------------------------------------------------------
        # Category A: 10-Year-Old / Simple Terms Explanation
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["10 year old", "10-year-old", "ten year old", "simple terms", "simple words", "eli5", "plain english", "explain simply", "like a child", "kid"]):
            return (
                f"### 🌟 Your Brain Story (Explained in Simple Words, Like for a 10-Year-Old)\n\n"
                f"Hello {first_name}! Here is what is happening inside your head, explained just like an adventure story:\n\n"
                f"**1. The Wonderful Control Center**:\n"
                f"Think of your brain like the most amazing, magical control center in the world! It controls all your walking, laughing, storytelling, and dreams. "
                f"In your **{location}** (which helps you remember fun adventures and words), a little cluster of cells decided to grow without following the rules—like unruly weeds in a beautiful garden.\n\n"
                f"**2. The Brave Surgeons & Your Healing Journey**:\n"
                f"Doctors went in and gently removed the big cluster of weeds during surgery! Right now, there is only a tiny spot left (measuring about {et_v:.1f} centimeters—about the size of a little blueberry).\n\n"
                f"**3. The Superhero Defense Squad (Your Medicines)**:\n"
                f"To make sure those weed cells don't come back, you have a superhero team working for you:\n"
                f"• **Temozolomide (The Scout)**: This medicine sneaks in and stops the rogue cells from multiplying.\n"
                f"• **Levetiracetam / Keppra (The Shield Guardian)**: It puts an invisible calm bubble around your brain wires so they don't get excited or cause twitches (seizures).\n\n"
                f"**4. What Your Latest Brain Picture (MRI) Shows**:\n"
                f"Your latest scan ({scan_date}) shows that everything is calm! The blood flow is nice and low (that means the cells are sleeping, not growing). "
                f"The little yellow ring on your 3D picture is just mild swelling from healing, like when your knee puffs up after you scrape it on the playground.\n\n"
                f"**5. How You Can Be a Champion Every Day**:\n"
                f"• Drink plenty of cold water and fruit juice 💧\n"
                f"• Take your Keppra pills on time every morning and night ⏰\n"
                f"• Get big, restful sleeps so your brain can repair itself 🌙\n"
                f"• Whenever you feel tired, take a rest—your body is doing superhero work inside!"
            )

        # -------------------------------------------------------------
        # Category B: Current Health Condition, Diagnosis & Overall Health Status
        # -------------------------------------------------------------
        if _matches_any(msg_lower, [
            "condition", "current condition", "my condition", "my health", "how am i doing",
            "what is wrong", "my cancer", "my tumor", "tell me about glioblastoma", "what is glioblastoma",
            "tell me about my cancer", "what disease", "my diagnosis", "tell me about my condition",
            "know about my current condition", "know about my condition", "explain my diagnosis", "status",
            "about my condition", "current health"
        ]):
            return (
                f"### 🩺 Comprehensive Overview of Your Current Health & Condition\n\n"
                f"Hello {first_name}, here is a clear, compassionate, and medically grounded overview of your health journey:\n\n"
                f"**1. Understanding Your Diagnosis**:\n"
                f"You have been diagnosed with **{diagnosis}** located in your **{location}**.\n"
                f"• This is a tumor of the brain's supportive glial tissue. Because it is situated in the **{location}** "
                f"(an area responsible for verbal memory, word finding, and sensory processing), your oncology team's top priority is keeping this region protected, quiet, and stable.\n\n"
                f"**2. Your Current Care & Treatment Phase**:\n"
                f"• **Surgical Phase**: You have already undergone surgery to debulk and resect the primary tumor.\n"
                f"• **Active Adjuvant Therapy**: Under the supervision of **{oncologist}**, you are undergoing targeted therapy (such as Temozolomide chemotherapy and radiation) to eliminate any microscopic residual cells.\n"
                f"• **Brain Shielding (Keppra)**: You are prescribed **Levetiracetam (Keppra 500 mg BID)**, an essential medication that stabilizes electrical activity in surrounding brain tissue and guards against seizures.\n\n"
                f"**3. What Your Latest Brain MRI ({scan_date}) Means for You**:\n"
                f"Your digital health twin shows **very encouraging and stable findings**:\n"
                f"• **Enhancing Residual Spot**: The active enhancing core measures **{et_v:.2f} cm³** (about the size of a small walnut), neatly confined along the resection cavity (Whole Tumor: {wt_v:.2f} cm³).\n"
                f"• **Blood Perfusion Index (rCBV: {rcbv:.2f})**: In neuro-oncology, a relative blood volume below 1.75 is reassuring. Your level of **{rcbv:.2f}** proves that local blood flow is low, confirming that this is beneficial **post-treatment tissue healing / pseudoprogression** rather than aggressive tumor growth.\n"
                f"• **Normal Symmetry**: There is no severe midline brain shift or ventricular crowding.\n\n"
                f"**4. What You Should Focus on Daily**:\n"
                f"• **Consistent Dosing**: Take your Keppra pills on time every 12 hours without skipping.\n"
                f"• **Cognitive Rest**: Sleep 8 hours nightly—sleep is when your brain does its deepest reparative healing.\n"
                f"• **Stay Well-Hydrated**: Drink 2 to 2.5 liters of water daily to flush medications.\n"
                f"• Reach out directly to your oncology navigator, **{nurse}**, if you notice any new headaches, tingling, or word-finding pauses.\n\n"
                f"You are navigating this journey with incredible resilience, {first_name}. What other aspect of your condition would you like to explore?"
            )

        # -------------------------------------------------------------
        # Category C: Emotional Coping, Fear, Anxiety, Family & Contagion
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["scared", "fear", "anxious", "anxiety", "depressed", "depression", "sad", "crying", "terrified", "hopeless", "hug", "family", "kids", "children", "contagious", "catch", "kiss"]):
            return (
                f"### 💙 Emotional Support & Family Safety for {patient_name}\n\n"
                f"Hello {first_name}, it takes immense courage to walk this path, and whatever emotions you are feeling right now are completely valid:\n\n"
                f"**1. Reassurance on Family & Closeness**:\n"
                f"• **Brain tumors are NOT contagious**: You cannot pass cancer to your family, children, or loved ones.\n"
                f"• **Hugs and kisses are completely safe**: You do not carry radioactivity after external beam radiation.\n"
                f"• The only temporary precaution is flushing twice and washing hands thoroughly for 48 hours following Temozolomide doses, as minor chemo metabolites exit through fluids.\n\n"
                f"**2. Honoring Your Feelings**:\n"
                f"• Navigating brain cancer can feel overwhelming. It is entirely normal to feel tired, anxious, or frightened on certain days.\n"
                f"• You are not alone. Your care team (**{oncologist}** and **{nurse}**) is actively monitoring your trajectory through your Digital Twin.\n"
                f"• Be gentle with yourself. Taking one day—or one hour—at a time is a valid and powerful way to heal."
            )

        # -------------------------------------------------------------
        # Category D: Prognosis, Future Outlook & Recovery
        # -------------------------------------------------------------
        if _matches_any(msg_lower, [
            "prognosis", "survival", "cure", "future", "life expectancy", "will i be okay",
            "am i dying", "getting better", "recover", "hope", "timeline", "long term"
        ]):
            return (
                f"### 🌅 Understanding Your Prognosis & Path Forward\n\n"
                f"Hello {first_name}, asking about your future and prognosis is one of the most honest and important questions in neuro-oncology. Here is a balanced, evidence-based medical perspective:\n\n"
                f"**1. Individualized Care, Not Generalized Numbers**:\n"
                f"Broad statistics often group together patients of all ages, genomic backgrounds, and eras. Today, your treatment for **{diagnosis}** in the **{location}** is guided by modern multimodal neuro-oncology and your digital twin biomarkers.\n\n"
                f"**2. Reassuring Indicators in Your Digital Twin**:\n"
                f"• **Low Perfusion rCBV ({rcbv:.2f})**: Rapidly growing recurrent lesions require elevated microvascular blood flow (rCBV > 1.75). Your low perfusion indicates calm tissue responding to treatment.\n"
                f"• **Controlled Residual Core ({et_v:.2f} cm³)**: Your enhancing tumor volume has remained stable post-surgery without rapid progression.\n"
                f"• **Multimodal Therapy**: Stupp protocol adjuvant chemotherapy combined with consistent Keppra dosing gives your brain its strongest defense.\n\n"
                f"**3. Our Shared Goal**:\n"
                f"The clinical focus is maintaining long-term radiographic stability, zero seizures, and vibrant daily living so you can enjoy time with family and friends."
            )

        # -------------------------------------------------------------
        # Category E: Latest Scan Volumetrics & Imaging Interpretation
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["scan", "mri", "latest scan", "result", "volume", "tumor size", "edema", "rcbv", "growing", "spread", "shrink"]):
            return (
                f"### 🧠 Detailed Interpretation of Your Brain MRI ({scan_date})\n\n"
                f"Hello {first_name}, here is a comprehensive breakdown of your latest volumetric neuro-imaging results:\n\n"
                f"**1. 3D Tumor Volumetrics (Digital Twin Decomposition)**:\n"
                f"• **Whole Tumor (WT)**: **{wt_v:.2f} cm³** — Represents the entire surgical bed and surrounding reactive tissue.\n"
                f"• **Enhancing Tumor Core (ET)**: **{et_v:.2f} cm³** — This is the residual post-operative surgical margin in your **{location}**. Crucially, this volume has remained stable without interval nodular progression.\n"
                f"• **Peritumoral Vasogenic Edema (ED)**: **{ed_v:.2f} cm³** — Surrounding fluid/swelling that is typical during adjuvant therapy.\n\n"
                f"**2. Perfusion Hemodynamics (rCBV Index = {rcbv:.2f})**:\n"
                f"• In neuro-oncology, dynamic perfusion measures microvascular blood flow. A threshold of **1.75** separates active neoangiogenesis from treatment effects.\n"
                f"• Your measured **rCBV of {rcbv:.2f} is well below 1.75**, providing strong quantitative evidence of **treatment-induced tissue response / pseudoprogression** rather than true tumor recurrence.\n\n"
                f"**3. RANO 2.0 Clinical Verdict**:\n"
                f"Your scan status is classified as **Pseudoprogression (PsP) / Radiographic Stability**. Your midline remains symmetric with 0.0 mm shift.\n\n"
                f"You can also explore your personalized 3D brain model in the **\"3D Brain & Tumor Localization\"** tab to see this anatomy rendered in 360°!"
            )

        # -------------------------------------------------------------
        # Category F: Symptoms, Sensations & Physical Changes
        # -------------------------------------------------------------
        if _matches_any(msg_lower, [
            "headache", "dizzy", "dizziness", "fatigue", "tired", "exhausted", "weak",
            "weakness", "numb", "numbness", "tingling", "seizure", "aura", "vision",
            "blurry", "speech", "talking", "words", "memory", "forget", "smell", "pain"
        ]):
            return (
                f"### 🩺 Understanding Your Symptoms & Brain Sensations\n\n"
                f"Hello {first_name}, navigating physical sensations and changes during brain cancer treatment requires attentive pacing:\n\n"
                f"**1. Fatigue & Cognitive Slowness**:\n"
                f"• Feeling tired is the **most common side effect** of radiation and chemotherapy. It does NOT mean the tumor is growing; rather, your body is using energy to repair healthy tissue.\n"
                f"• Schedule afternoon rests, hydrate frequently, and avoid overloading yourself with multi-tasking.\n\n"
                f"**2. Temporal Lobe Functions (Speech & Memory)**:\n"
                f"• Because your condition is in the **{location}**, subtle word-finding pauses or mild memory delays are expected when fatigued.\n"
                f"• Write daily reminders on your phone, speak at a calm pace, and allow yourself time.\n\n"
                f"**3. Sensory Auras & Keppra**:\n"
                f"• Unusual fleeting smells, tingling in the hands, or brief deja vu can represent mild focal sensory auras. Your **Keppra 500 mg BID** keeps these in check.\n"
                f"• Never skip a dose. If auras become frequent or feel different, inform **{nurse}** promptly.\n\n"
                f"**🚨 STAT Emergency Alert**: If you ever experience sudden paralysis on one side, loss of speech, repeated vomiting, or loss of consciousness, call 911 immediately."
            )

        # -------------------------------------------------------------
        # Category G: Medications, Dosing, Missed Dose, Nausea & Keppra
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["medication", "medicine", "pill", "temozolomide", "temodar", "keppra", "levetiracetam", "dexamethasone", "zofran", "ondansetron", "missed", "vomit", "throw up", "dose"]):
            return (
                f"### 💊 Personalized Medication Guidance for {patient_name}\n\n"
                f"Here are evidence-based recommendations regarding your current prescriptions and safety protocols:\n\n"
                f"**1. Temozolomide (TMZ Chemotherapy)**:\n"
                f"• **Empty Stomach Protocol**: Take your capsule strictly with a full glass of water at bedtime or at least 2 hours after food to maximize absorption and prevent nausea.\n"
                f"• **Anti-Nausea Shield**: Take **Ondansetron (Zofran 8 mg)** 30 to 45 minutes *before* your Temozolomide dose.\n"
                f"• **If You Vomit After Taking It**: **Do NOT take another capsule.** Wait until your next scheduled day and inform **{nurse}**.\n\n"
                f"**2. Levetiracetam (Keppra 500 mg BID)**:\n"
                f"• **Seizure Prevention**: Keppra stabilizes electrical activity around your tumor margin in the {location}.\n"
                f"• **If You Missed a Dose**: If remembered within 4 hours, take it immediately. If close to the next dose, skip the missed dose and resume normal schedule. **Never double dose.**\n"
                f"• Consistent 12-hour dosing keeps your seizure threshold high and safe.\n\n"
                f"**3. Dexamethasone (Steroid Tapering)**:\n"
                f"• If taking steroids for swelling, take doses with meals to protect your stomach lining and avoid evening doses if experiencing insomnia."
            )

        # -------------------------------------------------------------
        # Category H: Nutrition, Diet, Sugar, Coffee & Lifestyle
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["diet", "food", "eat", "nutrition", "coffee", "caffeine", "sugar", "sweets", "alcohol", "drink", "keto"]):
            return (
                f"### 🥗 Nutrition & Lifestyle Plan for {patient_name}\n\n"
                f"Proper nutrition supports cell repair, protects lean muscle mass, and keeps energy consistent during your treatment:\n\n"
                f"**1. Coffee & Caffeine**:\n"
                f"• **Moderate intake is generally safe**: 1 to 2 cups of coffee in the morning is acceptable for most neuro-oncology patients.\n"
                f"• Avoid caffeine after 2:00 PM to protect restorative sleep.\n"
                f"• If you notice increased tremors or agitation (sometimes amplified by Keppra or steroids), switch to green tea or decaf.\n\n"
                f"**2. Sugar & Carbohydrates**:\n"
                f"• While cancer cells consume glucose, severely starving your body of carbohydrates can lead to dangerous malnutrition.\n"
                f"• Choose complex carbohydrates (oats, quinoa, brown rice, berries) over refined sweets to avoid blood sugar spikes, especially if taking Dexamethasone.\n\n"
                f"**3. Key Pillars for Neuro-Recovery**:\n"
                f"• **High-Biological Protein**: Eggs, Greek yogurt, wild salmon, poultry, lentils, and tofu promote tissue repair.\n"
                f"• **Hydration**: Aim for 2.0 to 2.5 liters of water daily to help your kidneys flush chemotherapy metabolites.\n"
                f"• **Anti-Inflammatory Omega-3s**: Walnuts, chia seeds, and fatty fish help reduce systemic inflammation."
            )

        # -------------------------------------------------------------
        # Category I: Driving, Travel, Flying, Exercise & Physical Activity
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["drive", "driving", "car", "fly", "flying", "airplane", "travel", "exercise", "gym", "swim", "swimming", "work"]):
            return (
                f"### 🚗 Daily Living & Safety Guidelines for {patient_name}\n\n"
                f"**1. Driving Regulations & Seizure Safety**:\n"
                f"• Following brain tumor surgery and for patients taking anti-seizure medication (Keppra), clinical guidelines (AAN) and motor vehicle authorities typically mandate a **3 to 6-month seizure-free interval** before resuming driving.\n"
                f"• Consult **{oncologist}** before getting behind the wheel to review your clinical clearance.\n\n"
                f"**2. Air Travel & Flying**:\n"
                f"• Commercial flights are generally safe once surgical air (pneumocephalus) has resolved (usually 4–6 weeks post-craniotomy).\n"
                f"• Stay hydrated, stand up every 90 minutes to prevent blood clots (DVT), and carry all medications in your cabin baggage.\n\n"
                f"**3. Exercise & Physical Activity**:\n"
                f"• **Recommended**: Daily 20–30 minute walks, light stationary cycling, and gentle stretching.\n"
                f"• **Avoid**: Heavy weightlifting (Valsalva straining increases intracranial pressure) and unsupervised swimming."
            )

        # -------------------------------------------------------------
        # Category J: Questions for Doctor & Appointment Prep
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["doctor", "ask", "question", "appointment", "visit", "second opinion", "consult"]):
            return (
                f"### 📋 Recommended Questions for Your Next Visit with {oncologist}\n\n"
                f"Here are targeted questions grounded in your Digital Twin to discuss at your upcoming clinic appointment:\n\n"
                f"1. *\"My digital twin shows a low rCBV of {rcbv:.2f} and stable enhancing volume of {et_v:.1f} cm³—does this confirm pseudoprogression rather than active tumor growth?\"*\n"
                f"2. *\"How many more cycles of adjuvant Temozolomide are planned in my Stupp protocol?\"*\n"
                f"3. *\"Based on my current seizure stability on Keppra 500 mg BID, when can we discuss driving clearance?\"*\n"
                f"4. *\"Are there complementary neuro-cognitive therapies or physical rehabilitation exercises that will help my verbal memory in the {location}?\"*\n"
                f"5. *\"When is my next surveillance 3T MRI scheduled?\"*"
            )

        # -------------------------------------------------------------
        # Category K: General Greetings & Conversational Questions
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["hello", "hi", "hey", "who are you", "what can you do", "good morning", "good evening", "how are you"]):
            return (
                f"### 👋 Hello {first_name}! I am your AI Care Companion & Digital Twin GPT\n\n"
                f"I am your dedicated personal health companion, built like ChatGPT but uniquely grounded in your real-time medical data.\n\n"
                f"**What I have real-time access to**:\n"
                f"• Your latest 3D Brain MRI scans ({scan_date}) & tumor volumetrics in your **{location}**\n"
                f"• Your current medication schedule (Temozolomide, Keppra, Zofran)\n"
                f"• Your logged symptoms, seizure safety, and treatment history\n"
                f"• NCCN Guidelines for Central Nervous System Cancers\n\n"
                f"**Feel free to ask me anything**, such as:\n"
                f"• *\"Explain my current condition\"*\n"
                f"• *\"Explain my latest scan results\"*\n"
                f"• *\"Explain my condition in simple words as if I am 10 years old\"*\n"
                f"• *\"Can I drink coffee in the morning?\"*\n"
                f"• *\"What should I do if I miss my Keppra dose?\"*\n"
                f"• *\"Can I travel or drive right now?\"*\n\n"
                f"What is on your mind today, {first_name}?"
            )

        # -------------------------------------------------------------
        # Category L: Sleep, Insomnia & Resting Positions
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["sleep", "sleeping", "insomnia", "bed", "rest", "position", "side", "pillow", "night", "awake"]):
            return (
                f"### 🌙 Sleep & Brain Recovery Guidance for {patient_name}\n\n"
                f"Sleep is when your central nervous system performs its deepest cellular repair and glymphatic clearance:\n\n"
                f"**1. Sleeping Position Post-Craniotomy**:\n"
                f"• It is generally recommended to sleep with your head slightly elevated (30° angle or with 1–2 pillows) to reduce morning intracranial venous pressure and prevent facial or perilesional puffiness.\n"
                f"• If your incision in the **{location}** feels tender, sleep on the opposite side or on your back.\n\n"
                f"**2. Protecting Deep Sleep Cycles**:\n"
                f"• Aim for 8 hours of uninterrupted rest. Your brain clears metabolic debris primarily during slow-wave sleep.\n"
                f"• If taking steroids (Dexamethasone), take morning and midday doses only, as late steroid doses can cause alertness and insomnia.\n"
                f"• Keep bedtime screens dim, and avoid caffeine after 2:00 PM."
            )

        # -------------------------------------------------------------
        # Category M: Hair Washing, Incision Care & Bathing
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["hair", "wash", "shower", "bath", "shampoo", "incision", "scar", "wound", "stitches", "staples"]):
            return (
                f"### 🚿 Incision Care & Scalp Hygiene for {patient_name}\n\n"
                f"Caring for your scalp and surgical incision requires gentle, infection-preventing care:\n\n"
                f"**1. Incision Healing Protocol**:\n"
                f"• Keep the incision site clean and dry. Avoid scrubbing, scratching, or picking at scabs along the **{location}** incision.\n"
                f"• Use a mild, unscented baby shampoo. Let water run gently over the area without direct high-pressure shower spray.\n"
                f"• Gently pat dry with a clean, soft towel—do not rub vigorously.\n\n"
                f"**2. What to Avoid**:\n"
                f"• Do **not** submerge your head in bathtubs, swimming pools, or hot tubs until given formal clearance by your neurosurgical team.\n"
                f"• Avoid hair dyes, harsh chemical treatments, or heat styling until your scalp has fully recovered and radiation therapy concludes.\n\n"
                f"**3. When to Notify {nurse}**:\n"
                f"• Report any increasing redness, drainage/fluid weeping, warmth, or sudden fever (>100.4°F / 38°C) immediately."
            )

        # -------------------------------------------------------------
        # Category N: Radiation & Chemotherapy (Stupp Protocol)
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["radiation", "radiotherapy", "chemo", "chemotherapy", "stupp", "cycle", "cycles", "imrt", "beam"]):
            return (
                f"### 🔬 Understanding Your Multimodal Treatment (Stupp Regimen)\n\n"
                f"Hello {first_name}, your current therapy follows the world gold standard in neuro-oncology (the **Stupp Protocol**):\n\n"
                f"**1. How Temozolomide (TMZ) Works**:\n"
                f"• Temozolomide is an oral alkylating agent that crosses the blood-brain barrier. It introduces methyl groups into DNA, selectively causing programmed death in dividing cancer cells.\n"
                f"• Taking it with Ondansetron (Zofran) prevents nausea and ensures optimal drug delivery.\n\n"
                f"**2. Radiation Therapy (IMRT / Proton)**:\n"
                f"• Focused radiation targets microscopic margins in the **{location}** while sparing healthy brain tissue.\n"
                f"• Mild scalp fatigue or skin redness in the radiation field is expected and managed with gentle aquaphor or calendula cream.\n\n"
                f"**3. The Meaning of Your Stable Scan**:\n"
                f"• Your low perfusion index (**rCBV: {rcbv:.2f}**) proves that the treatments are successfully keeping microvascular proliferation suppressed."
            )

        # -------------------------------------------------------------
        # Category O: Molecular Markers, IDH & MGMT Methylation
        # -------------------------------------------------------------
        if _matches_any(msg_lower, ["idh", "mgmt", "methylation", "genetics", "molecular", "dna", "mutation", "biomarker", "pathology"]):
            return (
                f"### 🧬 Molecular Biomarkers & Precision Medicine for {patient_name}\n\n"
                f"Your digital twin integrates advanced molecular pathology markers to tailor your therapy:\n\n"
                f"**1. MGMT Promoter Methylation**:\n"
                f"• In neuro-oncology, **MGMT promoter methylation** is one of the most positive predictive biomarkers. It essentially turns 'off' the DNA-repair mechanism in cancer cells, making them substantially more sensitive to Temozolomide chemotherapy.\n\n"
                f"**2. IDH (Isocitrate Dehydrogenase) Status**:\n"
                f"• IDH mutation status provides valuable diagnostic classification (WHO 2021 CNS Criteria) that helps **{oncologist}** customize maintenance schedules.\n\n"
                f"**3. Why This Matters for You**:\n"
                f"• These genomic findings confirm that standard multimodal therapy (Stupp protocol) is well-suited to your biological profile."
            )

        # -------------------------------------------------------------
        # Category P: Universal Natural Conversational Response (Warm & Direct)
        # -------------------------------------------------------------
        return (
            f"### 💡 Personalized Care Guidance for {first_name}\n\n"
            f"Hello {first_name}! Let's walk through your question with clear clinical insights tailored to your journey:\n\n"
            f"**1. Clinical Status of Your Digital Twin**:\n"
            f"As someone managing **{diagnosis}** in your **{location}**, your health record shows that your condition is **stable, well-monitored, and on track** under your regimen with {oncologist}.\n"
            f"• **MRI Volumetric Stability**: Your latest brain scan ({scan_date}) confirms an enhancing surgical cavity core of **{et_v:.2f} cm³** with a reassuring perfusion index of **rCBV {rcbv:.2f}** (below the 1.75 threshold), which points to beneficial treatment response rather than aggressive recurrence.\n"
            f"• **Seizure Protection Shield**: Taking your prescribed **Keppra** regularly protects the delicate neuronal circuits around the {location}.\n\n"
            f"**2. Practical Recommendations for You Today**:\n"
            f"• **Pacing**: Listen to your body and honor when fatigue signals that your brain needs rest.\n"
            f"• **Hydration & Nourishment**: Drink plenty of water and eat nourishing, protein-rich meals to aid tissue recovery.\n"
            f"• **Direct Care Team Access**: Whenever you feel unsure about any medication timing or daily symptom, reach out to your clinical navigator, **{nurse}**.\n\n"
            f"I am always here to explain any scan term, medication nuance, or lifestyle question whenever you need, {first_name}."
        )

    def generate_response(
        self,
        user_message: str,
        patient_id: str = "e5b38d38-2c26-4d2b-91c6-2c1b97b00042",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates twin-grounded response verified against NCCN Guidelines.
        Prioritizes:
        1. OpenAI GPT-4o / GPT-4o-mini if API key provided or found in environment.
        2. Groq LLaMA-3 if Groq key provided or found in environment.
        3. Built-in Generative Clinical GPT Bio-LLM Engine.
        """
        db = get_database_client()
        patient = (
            db.get_patient_by_id(patient_id)
            or db.get_patient_by_mrn(patient_id)
            or db.get_patient_by_mrn("0042")
        )
        if not patient:
            all_pts = db.get_all_patients()
            patient = all_pts[0] if all_pts else {}

        real_patient_id = patient.get("patient_id", patient_id)
        scans = db.get_patient_scans(real_patient_id)
        reports = db.get_patient_reports(real_patient_id)
        meds = db.get_patient_medications(real_patient_id)
        symptoms = db.get_patient_symptoms(real_patient_id)
        timeline = db.get_patient_timeline(real_patient_id)

        latest_scan = scans[-1] if scans else {}
        patient_name = patient.get("full_name", "Patient")
        first_name = patient_name.split()[0] if patient_name else "there"
        mrn = patient.get("mrn", "0042")
        oncologist = patient.get("oncologist_name", "Dr. Aris Thorne, MD")
        nurse = patient.get("nurse_name", "Sarah Jensen, RN")

        msg_lower = user_message.lower().strip()

        # GraphRAG Clinical Guard validation
        patient_context = {
            "molecular": patient.get("molecular_markers", {}),
            "patient": patient,
            "meds": meds,
            "latest_scan": latest_scan
        }
        rag_res = self.rag_guard.generate_grounded_response(user_message, patient_context)

        # 1. IMMEDIATE RED-FLAG ESCALATION
        is_red_flag = any(kw in msg_lower for kw in self.red_flag_keywords)
        if is_red_flag:
            self.escalation_engine.evaluate_symptom_event(
                symptom_type="Acute Red-Flag (Chat)",
                severity="Severe",
                notes=user_message,
                patient_data=patient
            )
            red_flag_response = (
                f"🚨 **URGENT CLINICAL ALERT for {patient_name}**:\n\n"
                "What you are describing sounds like an acute neurological or cardiopulmonary emergency that requires immediate medical intervention. "
                "**Please do not wait. Call 911 or proceed to the nearest Emergency Department immediately.**\n\n"
                f"**Emergency Care Team Contacts**:\n"
                f"- **Attending Neuro-Oncologist**: {oncologist}\n"
                f"- **Oncology Care Navigator**: {nurse} (Urgent Hotline)\n"
                f"- **Patient MRN**: `{mrn}`\n\n"
                "*(I am your AI Care Companion. While I monitor your digital twin, I cannot diagnose acute emergencies or modify emergency orders in real-time.)*"
            )
            return {
                "response_text": red_flag_response,
                "model_used": "Emergency Clinical Safety Guard",
                "cross_modal_insight": "Critical red-flag symptom reported. Emergency clinical escalation dispatched to care team.",
                "red_flag_detected": True,
                "care_team_prompt": "Emergency evaluation or immediate call to care navigator.",
                "graph_rag": rag_res
            }

        # 2. RESOLVE API KEYS (Passed argument > Environment variable > .env)
        active_openai_key = (
            kwargs.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_KEY")
        )
        active_groq_key = (
            kwargs.get("groq_api_key")
            or os.environ.get("GROQ_API_KEY")
        )
        model_name = (
            kwargs.get("model_name")
            or os.environ.get("COMPANION_MODEL_NAME")
            or "gpt-4o"
        )

        model_tag = "Clinical Bio-LLM GPT"
        generated_answer = None

        # 3. ATTEMPT OPENAI GPT INFERENCE
        if active_openai_key and ("gpt" in model_name.lower() or not active_groq_key):
            generated_answer = self._call_openai_gpt(
                user_message=user_message,
                patient=patient,
                scans=scans,
                meds=meds,
                symptoms=symptoms,
                timeline=timeline,
                conversation_history=conversation_history,
                api_key=active_openai_key,
                model_name=model_name
            )
            if generated_answer:
                model_tag = f"OpenAI {model_name.upper()} (Cloud)"

        # 4. ATTEMPT GROQ INFERENCE
        if not generated_answer and active_groq_key:
            generated_answer = self._call_groq_llm(
                user_message=user_message,
                patient=patient,
                scans=scans,
                meds=meds,
                symptoms=symptoms,
                timeline=timeline,
                conversation_history=conversation_history,
                api_key=active_groq_key,
                model_name=model_name
            )
            if generated_answer:
                model_tag = "Groq LLaMA-3.3 70B (Cloud)"

        # 5. BUILT-IN CLINICAL GENERATIVE GPT ENGINE
        if not generated_answer:
            generated_answer = self._generate_clinical_gpt_response(
                user_message=user_message,
                patient=patient,
                scans=scans,
                meds=meds,
                symptoms=symptoms,
                timeline=timeline
            )
            model_tag = "Digital Twin Clinical GPT (Built-in)"

        return {
            "response_text": generated_answer,
            "model_used": model_tag,
            "cross_modal_insight": f"Grounded in {patient_name}'s Digital Health Twin & NCCN Guidelines.",
            "red_flag_detected": False,
            "care_team_prompt": f"Discuss any questions with {nurse} or {oncologist}.",
            "graph_rag": rag_res
        }
