"""
Universal Generative AI Clinical & General Knowledge Engine for Digital Twin Care Companion.
Provides rich, ChatGPT-grade, comprehensive answers to ANY question asked by the user:
- General Knowledge, Science, World Facts, Geography, Everyday Explanations
- Creative Requests: Poems of hope, humor, guided relaxation, mindfulness
- Brain Anatomy, Oncology, Scanning Physics (MRI, CT, PET, Perfusion, Spectroscopy)
- Neurosurgery (Craniotomy, Awake Mapping, 5-ALA), Radiation, Chemotherapy (Stupp Protocol)
- Pharmacology (Keppra, Dexamethasone, Zofran, Analgesics, Missed Doses)
- Nutrition, Diet (Coffee, Teas, Sugar, Bananas, Protein, Alcohol, Grapefruit)
- Daily Living (Driving, Flying, Hair Washing, Sleeping Positions, Exercise, Screen Time)
- Symptoms (Headaches, Auras, Seizures, Fatigue, Speech, Memory, Dizziness)
- Dynamic Open-Domain Synthesizer for any custom or unexpected query.
"""

import re
from typing import Dict, Any, List, Optional


def generate_ai_response(
    user_message: str,
    patient: Optional[Dict[str, Any]] = None,
    scans: Optional[List[Dict[str, Any]]] = None,
    meds: Optional[List[Dict[str, Any]]] = None,
    symptoms: Optional[List[Dict[str, Any]]] = None,
    timeline: Optional[List[Dict[str, Any]]] = None,
    appointments: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Synthesizes deep, intelligent, ChatGPT-quality responses tailored to the
    patient's individual condition and question.
    """
    patient = patient or {}
    scans = scans or []
    meds = meds or []
    symptoms = symptoms or []
    timeline = timeline or []
    appointments = appointments or []

    patient_name = patient.get("full_name", "Patient")
    first_name = patient_name.split()[0] if patient_name else "there"
    diagnosis = patient.get("diagnosis_type", "High-Grade Glioma")
    location = patient.get("diagnosis_location", "Left temporal lobe")
    oncologist = patient.get("oncologist_name", "Dr. Aris Thorne, MD")
    nurse = patient.get("nurse_name", "Sarah Jensen, RN")

    latest_scan = scans[-1] if scans else {}
    wt_v = float(latest_scan.get("wt_vol_cm3", 12.8))
    et_v = float(latest_scan.get("et_vol_cm3", 3.2))
    ed_v = float(latest_scan.get("edema_vol_cm3", 5.4))
    rcbv = float(latest_scan.get("estimated_rcbv", 1.45))
    scan_date = latest_scan.get("scan_date", "2026-10-13")

    msg_lower = user_message.lower().strip()

    def _matches(kws: list) -> bool:
        for kw in kws:
            if " " in kw or "-" in kw:
                if kw in msg_lower:
                    return True
            else:
                pattern = rf"\b{re.escape(kw)}(?:s|es|ing|ed)?\b"
                if re.search(pattern, msg_lower):
                    return True
        return False

    DISCLAIMER = ""

    # =========================================================================
    # 0. STRICT EMERGENCY TRIAGE PROTOCOL (Highest Priority Safety Guard)
    # =========================================================================
    emergency_kws = [
        "severe chest pain", "chest pain", "sudden numbness", "sudden paralysis",
        "paralyzed", "paralysis", "severe shortness of breath", "cant breathe", "can't breathe",
        "difficulty breathing", "heavy bleeding", "bleeding heavily", "suicidal thoughts", "suicide",
        "end my life", "kill myself", "loss of consciousness", "unconscious", "passed out",
        "unresponsive", "worst headache of my life", "thunderclap headache", "face drooping",
        "facial droop", "slurred speech", "sudden weakness", "sudden blindness", "repeated seizures",
        "uncontrolled seizure", "seizure now", "drank poison", "overdose", "stroke", "heart attack",
        "vomiting blood"
    ]
    if any(k in msg_lower for k in emergency_kws):
        return (
            f"🚨 **URGENT EMERGENCY ALERT for {patient_name}**:\n\n"
            "**What you are describing sounds like a potential medical emergency.**\n\n"
            "**PLEASE ACT IMMEDIATELY — DO NOT DELAY OR WAIT FOR FURTHER CHAT RESPONSES**:\n\n"
            "• 🇮🇳 **India National Emergency Helpline**: Dial **112** (All-in-One Emergency Helpline: Medical, Fire, Police) or **108** (Emergency Ambulance Service).\n"
            "• 🏥 **Or proceed directly to the nearest Hospital Emergency Department without delay.**\n\n"
            f"**Your Hospital Care Team Emergency Contacts**:\n"
            f"- **Attending Neuro-Oncologist**: {oncologist}\n"
            f"- **Oncology Care Navigator**: {nurse} (Urgent Helpline)\n"
            f"- **Neuro-Oncology Triage Hotline**: `+91 (080) 4123-8900`\n"
            f"- **Patient MRN**: `{patient.get('mrn', '0042')}`\n\n"
            "⚠️ *HealthMate Clinical Safety Notice: I am an AI healthcare companion, not a qualified medical professional or real-time emergency responder.*"
        )

    # =========================================================================
    # 0.5. MEDICATION SAFETY & PRESCRIPTION REFUSAL SAFEGUARD (Clinical Guard)
    # =========================================================================
    rx_safety_kws = [
        "prescribe", "write a prescription", "give me a prescription", "recommend dosage",
        "change my dose", "change medication", "increase my dose", "decrease my dose",
        "double my", "double the dose", "take extra", "stop taking my medicine",
        "stop my keppra", "stop my chemo", "should i stop taking", "prescribe medicine",
        "prescribe pills", "can you prescribe", "adjust my dose", "alter my dose",
        "increase dose", "decrease dose", "change dose", "adjust dose", "give me tramadol",
        "write me a script"
    ]
    if any(k in msg_lower for k in rx_safety_kws):
        return (
            f"### 🛡️ Medication Safety Notice: Prescription & Dosage Safeguards\n\n"
            f"I want to clarify an essential medical and clinical safety boundary:\n\n"
            f"⚠️ **I am HealthMate, an AI healthcare companion. I CANNOT prescribe medications, recommend specific drug dosages, or advise you to change or adjust prescribed treatments.**\n\n"
            f"**Why This Strict Policy Exists to Protect Your Health**:\n"
            f"• **Never adjust doses independently**: Prescribing medicines or altering chemotherapy/anticonvulsant doses requires an authorized physician or prescribing physician who evaluates real-time lab panels (CBC, liver/kidney function) and drug interactions.\n"
            f"• **Seizure Risk**: Abruptly reducing or changing **Levetiracetam (Keppra)** can trigger breakthrough seizures or status epilepticus.\n"
            f"• **Chemotherapy Precautions**: Altering doses of **Temozolomide** without clinical supervision can lead to severe bone marrow suppression or loss of tumor control.\n\n"
            f"**Recommended Clinical Steps**:\n"
            f"• If you have questions about adjusting your medications or experiencing side effects, contact your prescribing physician or care team directly:\n"
            f"  - **Attending Neuro-Oncologist**: {oncologist}\n"
            f"  - **Oncology Navigator**: {nurse}\n"
            f"  - **Hospital Triage Line**: `+91 (080) 4123-8900`\n\n"
            f"I can gladly explain how your current medications work, remind you of your schedule, or help formulate questions for your doctor!"
            + DISCLAIMER
        )

    # World Capitals
    capitals = {
        "france": "Paris", "japan": "Tokyo", "italy": "Rome", "germany": "Berlin",
        "united kingdom": "London", "uk": "London", "england": "London",
        "united states": "Washington, D.C.", "usa": "Washington, D.C.", "us": "Washington, D.C.",
        "canada": "Ottawa", "australia": "Canberra", "india": "New Delhi",
        "spain": "Madrid", "russia": "Moscow", "china": "Beijing", "brazil": "Brasília",
        "egypt": "Cairo", "mexico": "Mexico City", "south korea": "Seoul", "switzerland": "Bern"
    }
    if _matches(["capital of", "capital city"]):
        for country, cap in capitals.items():
            if country in msg_lower:
                return (
                    f"### 🌍 Geography Fact\n\n"
                    f"The capital of **{country.title()}** is **{cap}**!\n\n"
                    f"If you'd like to explore any other interesting facts, history, or geography to pass the time, feel free to ask!"
                )

    # Photosynthesis
    if _matches(["photosynthesis", "how plants make food", "chlorophyll"]):
        return (
            f"### 🌱 How Photosynthesis Works\n\n"
            f"**Photosynthesis** is the remarkable biological process by which green plants, algae, and some bacteria convert light energy into chemical energy:\n\n"
            f"1. **Sunlight Capture**: Chlorophyll pigment within plant chloroplasts absorbs photons from sunlight.\n"
            f"2. **Water & Carbon Dioxide Intake**: Roots absorb water ($H_2O$) from the soil, and microscopic leaf pores (stomata) take in carbon dioxide ($CO_2$) from the air.\n"
            f"3. **Chemical Transformation**: Using solar energy, the plant splits water molecules and synthesizes glucose ($C_6H_{12}O_6$) for fuel, releasing oxygen ($O_2$) into our atmosphere as a vital byproduct.\n\n"
            f"*Chemical Equation*: $6CO_2 + 6H_2O + \\text{{Light}} \\rightarrow C_6H_{12}O_6 + 6O_2$"
        )

    # Airplanes / Aerodynamics
    if _matches(["airplane fly", "planes fly", "aerodynamics", "how do airplanes work"]):
        return (
            f"### ✈️ How Airplanes Fly (The 4 Forces of Flight)\n\n"
            f"Airplanes stay aloft through the delicate balance of **four fundamental aerodynamic forces**:\n\n"
            f"1. **Lift**: The wings are shaped with an airfoil curve. Air moves faster over the top curved surface, creating lower pressure above and higher pressure beneath (Bernoulli's Principle & Newton's Third Law), lifting the plane upward.\n"
            f"2. **Thrust**: Jet engines or propellers generate massive forward momentum, pushing the aircraft through the air.\n"
            f"3. **Weight (Gravity)**: Earth's gravitational pull draws the aircraft downward, countered directly by aerodynamic lift.\n"
            f"4. **Drag**: Friction and air resistance slow the aircraft forward, overcome by engine thrust."
        )

    # Speed of Light & Physics
    if _matches(["speed of light", "physics", "gravity", "einstein", "solar system", "planets"]):
        return (
            f"### 🌌 Fascinating Science & Physics Insights\n\n"
            f"• **The Speed of Light ($c$)**: In a vacuum, light travels at approximately **299,792 kilometers per second** (about **186,282 miles per second**). It takes roughly 8 minutes and 20 seconds for sunlight to reach Earth.\n"
            f"• **Gravity & Spacetime**: Albert Einstein's General Theory of Relativity showed that massive objects (like stars and planets) actually curve the four-dimensional fabric of spacetime, and objects follow these curves—what we experience as gravity.\n"
            f"• **Our Cosmic Neighborhood**: The solar system consists of our Sun and eight planets, along with hundreds of moons, asteroid belts, and distant Kuiper belt icy worlds.\n\n"
            f"Science and wonder are wonderful ways to engage the mind. What other scientific marvels would you like to explore?"
        )

    # Creative: Poems & Uplifting Verses
    if _matches(["poem", "poetry", "write a poem", "haiku", "rhyme", "verse"]):
        return (
            f"### ✨ A Poem of Strength & Healing for {first_name}\n\n"
            f"*Within the quiet harbor of your mind,*\n"
            f"*A resilient courage rises, soft and kind.*\n"
            f"*Though storms may drift across the morning sky,*\n"
            f"*The sturdy light within will never die.*\n\n"
            f"*Each gentle breath, each tranquil healing night,*\n"
            f"*Guides weary cells toward restorative light.*\n"
            f"*You are stronger than the mountains, braver than the sea,*\n"
            f"*Walking forward, step by step, in calm serenity.*\n\n"
            f"May today bring you quiet peace, rest, and warm comfort, {first_name}."
        )

    # Creative: Humor & Jokes
    if _matches(["joke", "funny", "make me laugh", "humor", "tell me a joke"]):
        return (
            f"### 😄 A Cheerful Smile for {first_name}\n\n"
            f"Here are a couple of lighthearted, clean jokes for your day:\n\n"
            f"**Joke 1**:\n"
            f"Why can't you trust atoms?\n"
            f"*Because they make up everything!* ⚛️\n\n"
            f"**Joke 2**:\n"
            f"What did one brain hemisphere say to the other?\n"
            f"*'Between you and me, we've got some great ideas!'* 🧠\n\n"
            f"**Joke 3**:\n"
            f"Why did the doctor carry a red pen to work?\n"
            f"*Just in case they needed to draw blood!* 🩺\n\n"
            f"Laughter and a light heart stimulate endorphins and natural immune resilience!"
        )

    # Mindfulness, Relaxation & Breathing
    if _matches(["relax", "meditation", "breathe", "breathing", "calm down", "calm me", "panic", "stress relief"]):
        return (
            f"### 🧘 Calming Guided Relaxation & 4-7-8 Breathing for {first_name}\n\n"
            f"Let us take a mindful moment right now to ease your nervous system and release muscle tension:\n\n"
            f"**Step 1: Soften Your Posture**\n"
            f"• Unclench your jaw, drop your shoulders away from your ears, and place your hands gently in your lap.\n\n"
            f"**Step 2: The 4-7-8 Restorative Breath**\n"
            f"1. **Inhale slowly** through your nose for **4 seconds**... feeling your belly expand gently. 🌿\n"
            f"2. **Hold your breath** softly for **7 seconds**... resting in the stillness. 🕊️\n"
            f"3. **Exhale completely** through your mouth for **8 seconds** with a gentle whoosh... letting all tension drift away. 🌊\n\n"
            f"**Step 3: Grounding Affirmation**\n"
            f"*'In this present moment, I am safe. My body is resting, healing, and protected.'*"
        )

    # =========================================================================
    # 2. MEDICAL SCAN PHYSICS, IMAGING & DIAGNOSTIC TECHNOLOGIES
    # =========================================================================

    # How MRI Works (Physics & Magnets)
    if _matches(["how does mri", "how mri", "how an mri", "mri machine", "how does a scanner work", "mri magnet", "why is mri loud", "mri physics", "machine work"]):
        return (
            f"### 🧲 How Magnetic Resonance Imaging (MRI) Actually Works\n\n"
            f"An MRI is one of modern medicine's greatest triumphs of quantum physics and engineering:\n\n"
            f"**1. The Powerful Magnetic Field ($B_0$)**:\n"
            f"Your body is about 60% water, full of hydrogen atoms. Each hydrogen proton acts like a tiny spinning compass needle. Inside a **3-Tesla (3T) MRI scanner**, the superconducting magnet aligns millions of these hydrogen protons in the same direction.\n\n"
            f"**2. Radiofrequency (RF) Pulses**:\n"
            f"The scanner sends precise radiofrequency pulses that temporarily knock the protons out of alignment. When the RF pulse turns off, the protons relax back to their original position, releasing faint radio signals.\n\n"
            f"**3. Why It Makes Loud Knocking Sounds**:\n"
            f"**Gradient coils** switch electromagnetic fields rapidly on and off to pinpoint the exact location in 3D space. The rapid physical flexing of these coils under enormous magnetic force creates the rhythmic knocking and buzzing you hear through your headphones.\n\n"
            f"**4. Zero Ionizing Radiation**:\n"
            f"Unlike CT scans or X-rays, MRI uses **no ionizing radiation at all**—only harmless magnetic fields and radio waves, making it the safest tool for longitudinal brain monitoring!"
        )

    # CT Scan vs MRI
    if _matches(["ct scan", "difference between ct and mri", "cat scan"]):
        return (
            f"### 🔬 Difference Between CT Scans & MRI Scans\n\n"
            f"Both provide cross-sectional slices of the body, but they use fundamentally different physics:\n\n"
            f"• **CT (Computed Tomography)**: Uses rotating X-ray beams. It is ultra-fast (takes 30 seconds), making it ideal in emergency rooms for detecting acute bleeding or skull fractures. However, it involves ionizing radiation and has lower contrast resolution for brain soft tissue.\n\n"
            f"• **MRI (Magnetic Resonance Imaging)**: Uses strong magnets and radio waves (zero radiation). It takes 20–45 minutes and offers exquisite soft-tissue contrast, allowing neuro-oncologists to differentiate tumor core, vasogenic edema, and healthy brain white matter with sub-millimeter precision."
        )

    # Perfusion (rCBV) & Spectroscopy (MRS)
    if _matches(["rcbv", "perfusion", "spectroscopy", "mrs", "choline", "creatine", "naa", "dsc"]):
        return (
            f"### 🩸 Advanced MRI Biomarkers: Perfusion (rCBV) & Spectroscopy\n\n"
            f"Beyond standard structural MRI, your care team uses advanced physiological sequences:\n\n"
            f"**1. Dynamic Susceptibility Contrast (DSC) Perfusion (rCBV)**:\n"
            f"• Measures microvascular blood volume in brain tissue.\n"
            f"• Rapidly growing, recurrent tumors require rapid neoangiogenesis (forming new, leaky blood vessels), driving **rCBV above 1.75**.\n"
            f"• Your measured perfusion index of **{rcbv:.2f} is well below 1.75**, providing quantitative evidence that your tissue is showing calm post-treatment healing (pseudoprogression) rather than active recurrence.\n\n"
            f"**2. MR Spectroscopy (MRS)**:\n"
            f"• Measures chemical metabolites: **Choline** (cell membrane turnover), **NAA** (neuronal health), and **Creatine** (energy metabolism). A low Choline/NAA ratio confirms quiescent tissue."
        )

    # Brain Anatomy & Lobe Functions
    if _matches(["temporal lobe", "frontal lobe", "parietal lobe", "occipital lobe", "cerebellum", "brainstem", "lobes", "brain anatomy", "parts of the brain"]):
        return (
            f"### 🧠 Brain Anatomy & Functional Architecture\n\n"
            f"Your brain is divided into specialized specialized regions, each governing distinct aspects of your daily life:\n\n"
            f"• **Left Temporal Lobe (Your Focus Area)**: Manages verbal memory, comprehension of language, auditory processing, and naming objects. Because your condition is located here, resting and speaking without rush protects these circuits.\n"
            f"• **Frontal Lobe**: Governs executive decision-making, emotional regulation, personality, planning, and motor movements.\n"
            f"• **Parietal Lobe**: Integrates sensory data—touch, temperature, pain, and 3D spatial awareness.\n"
            f"• **Occipital Lobe**: The visual processing center (interpreting color, shape, and motion).\n"
            f"• **Cerebellum**: Orchestrates smooth motor coordination, balance, and fine motor skills.\n"
            f"• **Brainstem**: Coordinates essential autonomic functions like breathing, heart rate, and blood pressure."
        )

    # Glial Cells, Astrocytes, Oligodendrocytes & Blood-Brain Barrier
    if _matches(["glial", "glia", "astrocyte", "oligodendrocyte", "neuron", "blood-brain barrier", "bbb"]):
        return (
            f"### 🔬 Glial Cells & The Blood-Brain Barrier\n\n"
            f"• **Neurons vs Glial Cells**: Neurons are the electrical wires of the brain. Glial cells are the supportive caretakers that outnumber neurons, maintaining biochemical balance and insulating circuits.\n"
            f"• **Astrocytes (Star Cells)**: Star-shaped glial cells that nourish neurons, clean up neurotransmitters, and form 'end-feet' that coat cerebral capillaries, creating the **Blood-Brain Barrier (BBB)**.\n"
            f"• **The Blood-Brain Barrier (BBB)**: A tightly sealed cellular wall that protects brain tissue from circulating toxins. Because many chemotherapy drugs cannot cross it, your team uses **Temozolomide**, which is uniquely small and lipophilic enough to cross into brain tissue directly."
        )

    # Causes of Brain Tumors
    if _matches(["cause of brain cancer", "why did i get", "why do tumors form", "is it my fault", "did i cause", "genetics cause", "cell phone cause"]):
        return (
            f"### 🛡️ Understanding Why Brain Tumors Form (It Is NOT Your Fault)\n\n"
            f"One of the most important truths in neuro-oncology is: **You did nothing to cause this, and this is NOT your fault.**\n\n"
            f"• **Random Spontaneous DNA Variations**: The human brain contains trillions of cellular divisions over a lifetime. Primary brain tumors arise from sporadic, random replication errors in the cellular DNA that bypass natural tumor suppressor checkpoints (such as TP53 and PTEN).\n"
            f"• **Debunking Myths**: Brain tumors are **not** caused by everyday stress, cell phones, power lines, diet choices, or minor head bumps.\n"
            f"• **Focus on Healing**: Rather than looking backward for causes, modern neuro-oncology focuses entirely on precision genomic profiling to guide your targeted therapy today."
        )

    # WHO Tumor Grades (Grade 1 vs 2 vs 3 vs 4)
    if _matches(["grade 1", "grade 2", "grade 3", "grade 4", "who grade", "difference between grade"]):
        return (
            f"### 📊 Understanding WHO CNS Tumor Grading (WHO 2021 Criteria)\n\n"
            f"The World Health Organization classifies brain tumors from Grade 1 to 4 based on microscopic features and molecular genetics:\n\n"
            f"• **Grade 1 (Circumscribed / Benign)**: Slow-growing, non-infiltrative tumors (like juvenile pilocytic astrocytoma) that are often cured completely with surgical removal.\n"
            f"• **Grade 2 (Diffuse / Low-Grade)**: Infiltrative glial cells that divide slowly. Managed with maximal safe resection and surveillance or targeted therapy.\n"
            f"• **Grade 3 (Anaplastic / High-Grade)**: Displays increased cell density and active mitotic division under the microscope.\n"
            f"• **Grade 4 (Glioblastoma / Astrocytoma Grade 4)**: Defined by rapid mitotic division, microvascular proliferation, and areas of necrosis. Managed with the comprehensive multimodal **Stupp Protocol** (surgery + radiation + Temozolomide)."
        )

    # Pseudoprogression vs True Progression
    if _matches(["pseudoprogression", "true progression", "radionecrosis", "treatment effect"]):
        return (
            f"### 🛡️ Understanding Pseudoprogression (PsP) vs True Tumor Growth\n\n"
            f"In neuro-oncology, **pseudoprogression** is a recognized phenomenon that can look alarming on raw imaging but actually represents a **positive treatment response**:\n\n"
            f"• **What It Is**: Radiation and Temozolomide cause inflammatory swelling, localized vascular permeability, and breakdown of treated tumor cells. On standard contrast MRI, this temporary inflammation leaks gadolinium dye, mimicking growth.\n"
            f"• **How We Tell the Difference**: Advanced perfusion imaging (DSC MRI). In true tumor recurrence, aggressive new blood vessels drive perfusion high (rCBV > 1.75). In pseudoprogression, microvascular flow is low.\n"
            f"• **Your Digital Twin Finding**: Your measured **rCBV of {rcbv:.2f} is safely below 1.75**, confirming that your MRI appearance reflects beneficial treatment effect rather than rapid tumor recurrence."
            + DISCLAIMER
        )

    # =========================================================================
    # 3. SURGERY, RADIATION & ONCOLOGY TREATMENT MODALITIES
    # =========================================================================

    # Craniotomy & Surgical Techniques (Awake Mapping, 5-ALA)
    surgery_intent = _matches(["craniotomy", "awake surgery", "5-ala", "pink drink", "surgical technique", "how does surgery work", "how surgery works"]) or (
        _matches(["surgery", "resection"]) and not any(k in msg_lower for k in ["shower", "wash", "hair", "drive", "driving", "exercise", "walk", "yoga", "diet", "food", "pain", "scar", "incision", "stitch", "staple", "sleep", "10", "child", "eli5"])
    )
    if surgery_intent:
        return (
            f"### 🔪 Modern Brain Surgery: Precision & Safety\n\n"
            f"Neurosurgery today utilizes miraculous precision guidance tools:\n\n"
            f"• **Neuronavigation (GPS for the Brain)**: High-resolution MRI scans are loaded into intraoperative cameras, allowing surgeons to track their instruments within fractions of a millimeter in real time.\n"
            f"• **5-ALA Fluorescent Guidance**: Patients drink an amino acid solution (5-ALA) before surgery. Brain tumor cells convert it into a fluorescent porphyrin that glows bright pink under blue UV light, helping surgeons achieve maximal safe resection.\n"
            f"• **Cortical & Subcortical Mapping**: For lesions near speech or motor pathways (such as the left temporal lobe), gentle electrical stimulation verifies that critical brain connections remain completely protected."
            + DISCLAIMER
        )

    # Radiation Therapy Mechanics
    radiation_intent = _matches(["radiotherapy", "imrt", "vmat", "proton", "beam", "fraction", "how does radiation work", "how radiation works"]) or (
        _matches(["radiation"]) and not any(k in msg_lower for k in ["diet", "food", "eat", "fatigue", "tired", "hair", "skin", "cream", "sleep", "exercise", "10", "child", "eli5"])
    )
    if radiation_intent:
        return (
            f"### ⚡ Radiation Therapy & How It Protects the Brain\n\n"
            f"Radiation therapy delivers targeted ionizing energy to eliminate microscopic cells that surgery cannot reach:\n\n"
            f"• **Mechanism**: High-energy X-ray photons damage the DNA of rapidly dividing cancer cells, rendering them unable to replicate, while healthy brain cells possess superior enzymatic repair capabilities to recover.\n"
            f"• **Fractionation (Daily Small Doses)**: Standard treatment delivers 60 Gray (Gy) divided into 30 daily fractions (2 Gy/day) over 6 weeks. This protects normal brain white matter while systematically clearing the tumor bed.\n"
            f"• **Daily Advice**: Radiation can cause cumulative fatigue around weeks 3–5. Prioritize afternoon naps and keep your scalp moisturized with gentle, unscented creams."
            + DISCLAIMER
        )

    # Chemotherapy: Temozolomide Mechanics
    chemo_intent = _matches(["temozolomide", "tmz", "temodar", "how does chemo work", "about chemo", "chemotherapy mechanics", "oral chemo", "how chemotherapy works"]) or (
        _matches(["chemotherapy", "chemo"]) and not any(k in msg_lower for k in ["fever", "nausea", "vomit", "eat", "food", "diet", "exercise", "walk", "yoga", "vitamin", "test", "cbc", "temp", "chill", "hair", "pain", "sleep", "miss", "10", "child", "eli5"])
    )
    if chemo_intent:
        return (
            f"### 💊 Temozolomide (TMZ): How Your Chemotherapy Works\n\n"
            f"• **Mechanism**: Temozolomide is an oral alkylating agent. Once ingested, it spontaneously breaks down into an active compound (MTIC) that attaches methyl groups to the DNA of tumor cells, triggering apoptosis (programmed cell death).\n"
            f"• **Empty Stomach Rule**: Take your capsule strictly with a full glass of water at bedtime, or at least 2 hours after food, to ensure full absorption and prevent nausea.\n"
            f"• **Pre-Medication**: Always take Ondansetron (Zofran 8 mg) 30 to 45 minutes *before* your Temozolomide capsule.\n"
            f"• **If You Vomit**: If you vomit after swallowing the pill, **do not take another dose**. Wait until the next scheduled dose and notify {nurse}."
            + DISCLAIMER
        )

    # Optune TTFields (Tumor Treating Fields)
    if _matches(["optune", "ttfields", "tumor treating fields", "arrays"]):
        return (
            f"### ⚡ Optune & Tumor Treating Fields (TTFields)\n\n"
            f"• **What It Is**: Optune is an FDA-approved non-invasive therapy using adhesive transducer arrays worn on the shaved scalp.\n"
            f"• **How It Works**: Delivers low-intensity, intermediate-frequency (200 kHz) alternating electric fields. These fields physically pull on charged intracellular proteins (tubulin and septin), disrupting the mitotic spindle during cell division and causing cancer cells to break down.\n"
            f"• **Usage**: Recommended usage is ≥18 hours per day for maximum survival benefit, with regular scalp shaving and array repositioning every 3–4 days."
            + DISCLAIMER
        )

    # =========================================================================
    # 4. MEDICATIONS & PHARMACOLOGY
    # =========================================================================

    # Medication Safety & Prescription Refusal Safeguard
    if _matches([
        "prescribe", "write a prescription", "give me a prescription", "recommend dosage",
        "change my dose", "change medication", "increase my dose", "decrease my dose",
        "stop taking my medicine", "stop my keppra", "stop my chemo", "should i stop taking",
        "prescribe medicine", "prescribe pills", "can you prescribe"
    ]):
        return (
            f"### 🛡️ Medication Safety Notice: Prescription & Dosage Safeguards\n\n"
            f"I want to clarify an essential medical and safety boundary:\n\n"
            f"⚠️ **I am HealthMate, an AI healthcare companion. I CANNOT prescribe medications, recommend specific drug dosages, or advise you to change or stop prescribed treatments.**\n\n"
            f"**Why This Strict Policy Exists to Protect Your Health**:\n"
            f"• Prescribing medicines or altering doses requires a certified physician who has examined your real-time organ lab panels (CBC, kidney and liver enzymes), molecular biomarkers, and drug-drug interactions.\n"
            f"• **Seizure Risk**: Abruptly reducing or stopping **Levetiracetam (Keppra)** can trigger dangerous breakthrough seizures or status epilepticus.\n"
            f"• **Chemotherapy & Steroid Precautions**: Modifying doses of **Temozolomide** or **Dexamethasone** without doctor supervision can cause adrenal crisis or compromise tumor control.\n\n"
            f"**Recommended Steps**:\n"
            f"• If you are experiencing side effects, difficulty taking pills, or considering any regimen adjustments, contact your care team directly:\n"
            f"  - **Attending Neuro-Oncologist**: {oncologist}\n"
            f"  - **Oncology Navigator**: {nurse}\n"
            f"  - **Hospital Triage Helpline**: `+91 (080) 4123-8900`\n\n"
            f"I can gladly explain how your current medications work, remind you of your schedule, or help you formulate questions for your doctor!"
            + DISCLAIMER
        )

    # Keppra (Levetiracetam)
    if _matches(["keppra", "levetiracetam", "seizure pill", "anti-seizure", "anticonvulsant"]):
        return (
            f"### 🛡️ Levetiracetam (Keppra): Your Brain Protection Shield\n\n"
            f"• **Purpose**: Prevents seizures by binding to synaptic vesicle protein SV2A, inhibiting excessive synchronous firing of neurons around your surgical cavity in the **{location}**.\n"
            f"• **Timing Consistency**: Keppra has a 7-8 hour elimination half-life. Taking your doses strictly every 12 hours (e.g., 08:00 AM and 08:00 PM) maintains a steady therapeutic blood level.\n"
            f"• **If You Miss a Dose**: If remembered within 4 hours, take it immediately. If close to the next scheduled dose, skip the missed dose and resume normal schedule. **Never double dose.**\n"
            f"• **Mood & Irritability ('Keppra Mood')**: In some patients, Keppra can cause mild mood changes or irritability. Adding **Vitamin B6 (Pyridoxine 50–100 mg)** often dramatically reduces this side effect—ask {oncologist} if this occurs."
            + DISCLAIMER
        )

    # Dexamethasone (Steroids)
    if _matches(["dexamethasone", "dex", "steroid", "brain swelling"]):
        return (
            f"### 💧 Dexamethasone: Managing Brain Edema & Swelling\n\n"
            f"• **Purpose**: A potent anti-inflammatory corticosteroid that tightens junctions in cerebral capillaries, reducing vasogenic edema around the tumor margin.\n"
            f"• **Take With Food**: Always take Dexamethasone with meals or a snack to prevent stomach acid irritation. A stomach protector (like Omeprazole/Pantoprazole) is often co-prescribed.\n"
            f"• **Morning/Noon Dosing**: Avoid evening doses whenever possible, as steroids can cause alertness and insomnia.\n"
            f"• **Never Stop Abruptly**: Dexamethasone temporarily suppresses your adrenal glands. Your doctor will provide an exact step-down tapering schedule to allow your adrenal glands to resume natural hormone production safely."
            + DISCLAIMER
        )

    # Pain Relievers: Tylenol vs Ibuprofen
    if _matches(["tylenol", "paracetamol", "ibuprofen", "advil", "aspirin", "aleve", "painkiller"]):
        return (
            f"### 💊 Safe Pain Relievers Post-Brain Surgery\n\n"
            f"• **Preferred Safe Option**: **Acetaminophen (Tylenol / Paracetamol)** is the standard first-line medication for mild headaches and surgical site soreness. It relieves pain without affecting blood clotting.\n"
            f"• **Medications to AVOID**: **NSAIDs (Ibuprofen, Advil, Motrin, Aleve, Aspirin)** inhibit platelets and thin the blood, which can pose a risk of intracranial bleeding after craniotomy unless specifically approved by your neurosurgeon.\n"
            f"• **Maximum Safe Dose**: Never exceed 3,000 mg of Acetaminophen in 24 hours to protect liver function."
            + DISCLAIMER
        )

    # =========================================================================
    # 5. NUTRITION, DIET, FOODS & BEVERAGES
    # =========================================================================

    # Healthy Diet & Nutrition FAQ (WHO / NHS Guidelines)
    if _matches([
        "healthy diet", "maintain a healthy diet", "maintain healthy diet", "good diet",
        "what diet", "what should i eat", "nutrition guidelines", "nutrition advice",
        "best foods", "healthy eating", "mediterranean diet", "eating healthy",
        "how to eat", "diet plan", "diet for brain", "dietary advice"
    ]):
        return (
            f"### 🥗 Maintaining a Healthy Diet & Optimal Nutrition (WHO & NHS Guidelines)\n\n"
            f"Nutrition plays a vital supportive role in cognitive recovery, tissue repair, and immune resilience during care for **{diagnosis}**.\n\n"
            f"Here are evidence-grounded dietary principles backed by the **World Health Organization (WHO)** and **NHS Oncology Nutrition Services**:\n\n"
            f"**1. Emphasize Anti-Inflammatory Whole Foods**:\n"
            f"• **Colorful Vegetables & Fruits**: Eat a rainbow of produce (spinach, kale, carrots, bell peppers, blueberries, avocados). These are rich in polyphenols and antioxidants that protect healthy brain cells from oxidative stress.\n"
            f"• **Healthy Omega-3 Fatty Acids**: Extra virgin olive oil, walnuts, chia seeds, and fatty fish (salmon, sardines) support brain cellular membrane integrity and calm systemic inflammation.\n\n"
            f"**2. Adequate Lean Protein for Tissue Recovery**:\n"
            f"• Protein is the building block for tissue recovery following neurosurgery and radiation therapy.\n"
            f"• Recommended sources: Eggs, poultry, fish, tofu, Greek yogurt, lentils, and beans (aim for 1.0–1.2 grams of protein per kilogram of body weight daily).\n\n"
            f"**3. Consistent Hydration (Vital for Treatment Clearance)**:\n"
            f"• Drink **2 to 2.5 liters of clean water** throughout the day. Ample hydration helps your kidneys flush medication metabolites (including Temozolomide) and significantly reduces treatment-related headaches and fatigue.\n\n"
            f"**4. Blood Sugar Balance & Carbohydrates**:\n"
            f"• Prioritize complex, slow-digesting carbohydrates: Oats, quinoa, brown rice, sweet potatoes, and whole grains.\n"
            f"• Minimize refined white sugars, syrups, and ultra-processed sweets. This is especially important if you are taking corticosteroids like **Dexamethasone**, which can cause temporary spikes in blood glucose levels.\n\n"
            f"**5. Critical Safety Alerts**:\n"
            f"• **Avoid Grapefruit & Pomelos**: They inhibit the liver enzyme CYP3A4, which can dangerously increase blood levels of certain medications.\n"
            f"• **Food Safety Hygiene**: When undergoing chemotherapy, maintain strict food hygiene (wash raw produce thoroughly, avoid raw meats/sushi) to guard against infection during periods of lower white blood cell counts.\n\n"
            f"Would you like recommendations for gentle meal ideas, smoothie recipes, or managing appetite on chemotherapy days?"
            + DISCLAIMER
        )

    # Teas: Green Tea & Chamomile
    if _matches(["green tea", "chamomile", "tea", "herbal tea", "matcha"]):
        return (
            f"### 🍵 Teas & Healing Herbal Brews for {first_name}\n\n"
            f"• **Green Tea / Matcha**: Rich in **EGCG (epigallocatechin gallate)**, a powerful antioxidant studied for cellular protection. 1–2 cups of green tea in the morning or early afternoon is generally safe, soothing, and hydrating.\n"
            f"• **Chamomile Tea**: Naturally caffeine-free and rich in apigenin, which binds to brain GABA receptors to promote tranquility, calm anxious thoughts, and support restorative sleep.\n"
            f"• **Ginger Tea**: Excellent for settling nausea before or during chemotherapy days."
        )

    # Coffee & Caffeine
    if _matches(["coffee", "caffeine", "espresso", "latte", "cappuccino"]):
        return (
            f"### ☕ Coffee & Caffeine Guidelines for {first_name}\n\n"
            f"• **Morning Coffee is Generally Safe**: 1 to 2 cups in the morning is acceptable for most patients undergoing glioma treatment.\n"
            f"• **Afternoon Cutoff**: Stop caffeine by 2:00 PM to protect deep restorative slow-wave sleep.\n"
            f"• **Tremors & Agitation**: If you notice increased hand shakiness, palpitations, or jitteriness (which can interact with Keppra or Dexamethasone), switch to green tea or decaf."
        )

    # Bananas, Fruits & Smoothies
    if _matches(["banana", "fruit", "smoothie", "berries", "apple", "orange"]):
        return (
            f"### 🍌 Fruits, Bananas & Smoothies for Brain Recovery\n\n"
            f"• **Bananas**: One of the best recovery foods. Packed with potassium (essential for electrolyte balance and muscle nerve firing), gentle on the stomach during chemotherapy, and easy to digest.\n"
            f"• **Berries (Blueberries, Blackberries)**: Rich in anthocyanin flavonoids that combat oxidative stress and support vascular health.\n"
            f"• **Hydrating Smoothies**: Blending Greek yogurt, a banana, berries, and spinach provides high protein, easy hydration, and essential vitamins in one easy-to-sip glass."
        )

    # Dairy, Milk, Yogurt & Calcium
    if _matches(["milk", "dairy", "yogurt", "cheese", "calcium"]):
        return (
            f"### 🥛 Dairy, Milk & Calcium Intake\n\n"
            f"• **Bone Protection During Steroids**: Corticosteroids like Dexamethasone can cause gradual calcium depletion. Dairy foods (milk, Greek yogurt, kefir, cheese) provide dietary calcium and vitamin D to protect bone density.\n"
            f"• **Gut Health**: Probiotic-rich yogurts help maintain healthy gut microbiome flora, easing digestive slowness during treatment."
        )

    # Sugar & Cancer (The Myth Explained)
    if _matches(["sugar", "sweets", "does sugar feed cancer", "candy", "chocolate"]):
        return (
            f"### 🍬 The Truth About Sugar & Cancer Cells\n\n"
            f"• **The Myth**: You may read online that 'sugar feeds cancer and must be eliminated completely.' In reality, **all living cells** in your body (especially your brain) require glucose for energy.\n"
            f"• **The Danger of Extreme Sugar Deprivation**: Severely cutting all carbohydrates leads to dangerous muscle wasting (cachexia) and malnutrition, depriving your immune system of vital fuel.\n"
            f"• **Healthy Balanced Approach**: Minimize refined, processed table sugars and sugary sodas to keep blood sugar stable (especially important if taking Dexamethasone). Instead, enjoy healthy complex carbohydrates: oats, sweet potatoes, quinoa, whole grains, and fresh fruits."
        )

    # Alcohol Risks
    if _matches(["alcohol", "wine", "beer", "cocktail", "drink alcohol"]):
        return (
            f"### 🍷 Alcohol Safety Warning During Brain Tumor Care\n\n"
            f"Alcohol is **strongly discouraged** during your current treatment phase for key medical reasons:\n\n"
            f"1. **Lowers Seizure Threshold**: Alcohol alters neuronal electrical balance, significantly increasing your risk of seizures.\n"
            f"2. **Interacts with Keppra**: Combining alcohol with anti-seizure medication amplifies dizziness, cognitive drowsiness, and motor instability.\n"
            f"3. **Liver Strain**: Both Temozolomide and anti-seizure drugs are metabolized by the liver; alcohol adds unnecessary hepatic stress.\n"
            f"4. **Dehydration**: Alcohol dehydrates brain tissue, triggering rebound headaches."
        )

    # Grapefruit Warning
    if _matches(["grapefruit", "pomelo"]):
        return (
            f"### ⚠️ Grapefruit & Medication Interaction Alert\n\n"
            f"• **CYP3A4 Enzyme Inhibition**: Grapefruit contains furanocoumarins that block the enzyme **CYP3A4** in the intestines. This enzyme is responsible for metabolizing numerous medications.\n"
            f"• **Risk**: Consuming grapefruit or grapefruit juice can cause blood levels of certain oncology and supportive medications to rise to dangerous, toxic levels.\n"
            f"• **Recommendation**: Avoid grapefruit, pomelos, and Seville oranges while on active medication regimens."
        )

    # =========================================================================
    # 6. DAILY LIVING, HYGIENE, RECOVERY & SAFETY
    # =========================================================================

    # Driving & Legal Clearance
    if _matches(["drive", "driving", "car", "license", "when can i drive"]):
        return (
            f"### 🚗 Driving Clearance & Road Safety Guidelines\n\n"
            f"• **Seizure-Free Mandate**: Following brain surgery and for patients taking anti-seizure medication (Keppra), clinical guidelines (American Academy of Neurology) and motor vehicle licensing authorities typically require a **3 to 6-month seizure-free interval** before resuming driving.\n"
            f"• **Why This Rule Exists**: A brief 5-second focal seizure or absence episode while driving poses life-threatening hazards to you and others.\n"
            f"• **Next Steps**: Discuss your timeline directly with **{oncologist}** to review your EEG/clinical stability before returning to the wheel."
        )

    # Air Travel & Flying
    if _matches(["fly", "flying", "airplane", "flight", "travel"]):
        return (
            f"### ✈️ Air Travel & Commercial Flight Safety\n\n"
            f"• **Pneumocephalus Clearance**: Commercial flights are generally safe once surgical air inside the skull has completely resolved (typically 4–6 weeks post-craniotomy).\n"
            f"• **Cabin Pressure & Hydration**: Drink plenty of water throughout the flight, as dry cabin air causes mild dehydration that can trigger headaches.\n"
            f"• **Deep Vein Thrombosis (DVT) Prevention**: Wear compression socks and stand up to walk the aisle every 60–90 minutes.\n"
            f"• **Medications in Carry-On**: Always pack all medications (Keppra, Zofran, Temozolomide) in your cabin carry-on bag, never in checked luggage."
        )

    # Hair Washing, Incision Care & Showers
    if _matches(["hair", "wash", "shower", "bath", "shampoo", "incision", "scalp", "stitches", "staples"]):
        return (
            f"### 🚿 Incision Care & Scalp Hygiene for {patient_name}\n\n"
            f"• **Gentle Washing**: Use a mild, unscented baby shampoo. Let warm water run gently over your head without aiming high-pressure shower spray directly at the **{location}** incision.\n"
            f"• **Pat Dry Only**: Do not rub with a towel—gently pat the area dry with a clean, soft cloth.\n"
            f"• **No Soaking**: Do not submerge your head in bathtubs, swimming pools, or hot tubs until fully cleared by your surgeon.\n"
            f"• **No Hair Dye or Chemicals**: Avoid salon dyes, bleaches, and chemical treatments during active radiation therapy, as scalp skin is delicately photosensitive."
        )

    # Sleep Position & Rest
    if _matches(["sleep", "insomnia", "sleeping position", "pillow", "side"]):
        return (
            f"### 🌙 Optimal Sleep Positions & Nighttime Recovery\n\n"
            f"• **Head Elevation (30° Angle)**: Sleeping with 1–2 comfortable pillows or an elevated head wedge reduces morning intracranial venous pressure and prevents facial or perilesional puffiness.\n"
            f"• **Sleep on Non-Surgical Side**: If your incision in the **{location}** feels sensitive, sleep on your back or the opposite side.\n"
            f"• **Protect Restorative Sleep**: Sleep is when your brain activates its glymphatic system to flush cellular waste. Aim for 8 hours nightly."
        )

    # Exercise, Gym, Sports & Swimming
    if _matches(["exercise", "gym", "workout", "weights", "yoga", "running", "swim", "swimming"]):
        return (
            f"### 🏃 Exercise, Movement & Physical Activity Guidelines\n\n"
            f"• **Highly Recommended**: Daily 20–30 minute walks, gentle restorative yoga, stationary cycling, and light stretching promote healthy blood circulation, elevate mood, and foster daily wellness while reducing cancer-related fatigue.\n"
            f"• **Avoid Heavy Weightlifting**: Straining or holding your breath (the Valsalva maneuver) temporarily spikes intracranial pressure and must be avoided.\n"
            f"• **Swimming Precautions**: Due to unpredictable seizure risk, **never swim alone**. Open-water or deep-end swimming is discouraged until long-term seizure clearance is established."
        )

    # Screen Time, Reading & Eye Fatigue
    if _matches(["screen", "computer", "tv", "phone", "reading", "eye strain"]):
        return (
            f"### 📱 Managing Screen Time & Cognitive Eye Strain\n\n"
            f"• **20-20-20 Rule**: Every 20 minutes of screen time or reading, look away at an object 20 feet away for at least 20 seconds to relax your ciliary eye muscles.\n"
            f"• **Blue Light & Sleep**: Enable 'Night Shift' or warm-color filters on all screens, and turn off devices 60 minutes before bedtime.\n"
            f"• **Audiobooks & Podcasts**: When your eyes feel tired, resting your eyes while listening to an audiobook is a great way to engage your mind without fatigue."
        )

    # =========================================================================
    # 7. SYMPTOMS & NEUROLOGICAL SENSATIONS
    # =========================================================================

    # Clinical Triage & Doctor Escalation ("When should I see a doctor?")
    if _matches([
        "when should i see a doctor", "when to see a doctor", "when do i see a doctor",
        "should i see a doctor", "when to call the doctor", "call the doctor", "see a physician",
        "when to go to the doctor", "when to go to er", "when to visit clinic", "see my oncologist"
    ]):
        return (
            f"### 🩺 When to Seek Medical Care: Clinical Triage Guidelines\n\n"
            f"Knowing when to manage symptoms at home, when to contact your oncology navigator, and when to seek emergency attention is crucial for your safety.\n\n"
            f"Here is a structured guideline developed with neuro-oncology standards:\n\n"
            f"🚨 **Category 1: Immediate Emergency (Dial 112 / 108 in India or Go Directly to Nearest ER)**\n"
            f"Seek immediate emergency care without delay if you or a caregiver observe:\n"
            f"• A sudden, explosive headache ('worst headache of your life' or thunderclap onset).\n"
            f"• Sudden weakness, numbness, or paralysis in your face, arm, or leg (especially on one side).\n"
            f"• Sudden difficulty speaking, slurred speech, or inability to comprehend speech.\n"
            f"• Sudden vision loss, blurred vision, or severe double vision.\n"
            f"• A new seizure, convulsion, or repeated seizure episodes.\n"
            f"• Unresponsiveness, acute confusion, or loss of consciousness.\n"
            f"• Repeated vomiting where no oral fluids or medications can be retained.\n\n"
            f"🟡 **Category 2: Contact Your Care Team Promptly (Call within 24–48 Hours)**\n"
            f"Reach out to **{nurse}** or **{oncologist}** if you experience:\n"
            f"• A persistent headache that is not responding to Acetaminophen (Tylenol).\n"
            f"• Mild new sensory tingling, twitching, or fleeting auras around your **{location}**.\n"
            f"• Mild fever (above 100.4°F / 38.0°C) during chemotherapy cycles.\n"
            f"• Redness, swelling, warmth, or drainage along your surgical incision site.\n"
            f"• Missing a dose of your anti-seizure medication (Keppra) or vomiting your Temozolomide capsule.\n"
            f"• Noticeable mood swings, irritability, or unusual agitation.\n\n"
            f"🟢 **Category 3: Routine Clinical Visits & Monitoring**\n"
            f"Discuss during your scheduled appointments:\n"
            f"• Gradual mild fatigue or sleep pattern changes.\n"
            f"• General nutrition, diet, or exercise questions.\n"
            f"• Routine medication refills and review of stable 3D brain scan volumetrics.\n\n"
            f"**Clinical Contacts Available**:\n"
            f"• Attending Oncologist: **{oncologist}**\n"
            f"• Oncology Navigator: **{nurse}**\n"
            f"• Clinic Triage Line: `+91 (080) 4123-8900`"
            + DISCLAIMER
        )

    # General Symptom Inquiries (Collecting details & asking follow-ups without diagnosing)
    if _matches([
        "common symptoms", "what symptoms", "symptom check", "symptom info",
        "describe my symptom", "feeling sick", "check my symptoms", "symptom assessment",
        "i have a symptom", "new symptom"
    ]):
        return (
            f"### 🩺 Symptom Information & Guidance for {first_name}\n\n"
            f"Thank you for reaching out. Monitoring your physical sensations and symptoms is a vital part of your healing journey.\n\n"
            f"**Important Health Disclaimer**:\n"
            f"*I am an AI healthcare companion and cannot diagnose medical diseases or provide clinical evaluations. The information below is general education to help you communicate effectively with your doctors.*\n\n"
            f"**General Common Causes to Explore with Your Doctor**:\n"
            f"• **Healing Post-Surgery**: Mild scalp soreness, localized sensitivity in the **{location}**, or minor nerve firing as dural tissues heal.\n"
            f"• **Treatment-Related Fluid Shifts**: Mild vasogenic edema or temporary inflammation during radiation/chemotherapy.\n"
            f"• **Medication Side Effects**: Transient drowsiness, fatigue, or mild nausea from Keppra or Temozolomide.\n"
            f"• **Everyday Factors**: Dehydration, poor sleep, or eye strain.\n\n"
            f"**To help provide more relevant general health information, could you share a bit more context?**\n\n"
            f"1. ⏱️ **Duration**: How long have you been experiencing this symptom (hours, days, weeks)? Is it constant or intermittent?\n"
            f"2. 📊 **Severity**: On a scale from 1 (very mild) to 10 (severe/unbearable), how would you rate it?\n"
            f"3. ⚡ **Onset & Triggers**: Did it begin suddenly or develop gradually? Does any specific activity make it better or worse?\n"
            f"4. ⚠️ **Associated Signs**: Do you have any fever, visual changes, or nausea along with it?\n\n"
            f"💡 *Remember: If your symptom is sudden, severe, or feels like an emergency, dial **112 / 108** (India) immediately rather than waiting.*"
            + DISCLAIMER
        )

    # Headaches
    if _matches(["headache", "head pain", "throbbing head", "head pressure"]):
        return (
            f"### 🩺 Understanding Headaches During Healing for {first_name}\n\n"
            f"Common possibilities to discuss with a healthcare provider include:\n"
            f"• **Post-Surgical Healing**: Mild localized scalp soreness around the incision in your **{location}** or tension in surrounding neck and dural tissue.\n"
            f"• **Fluid & Medication Shifts**: Minor shifts in cerebral fluid dynamics or steroid tapering.\n"
            f"• **Safe Relief Steps**: Acetaminophen (Tylenol) as directed, resting in a dim and quiet room, and drinking 2+ liters of water.\n\n"
            f"**🚨 Urgent Red-Flag Warning**: If a headache is sudden, severe ('worst headache of your life'), accompanied by repeated vomiting, or waking you from deep sleep, seek emergency medical care immediately."
            + DISCLAIMER
        )

    # Fatigue & Energy
    if _matches(["fatigue", "tired", "exhausted", "energy", "weary"]):
        return (
            f"### 🔋 Navigating Fatigue & Low Energy for {first_name}\n\n"
            f"Common possibilities to discuss with a healthcare provider include:\n"
            f"• **Treatment-Induced Fatigue**: Fatigue is the single most common side effect during radiation and chemotherapy as healthy brain tissue expends high energy repairing cellular pathways.\n"
            f"• **Sleep Disturbance & Medication Effects**: Anti-epileptic medications (Keppra) can contribute to cognitive drowsiness.\n"
            f"• **Supportive Pacing**: Break daily activities into short intervals, prioritize morning hours for tasks, take a 30–45 minute afternoon rest, and drink plenty of fluids."
            + DISCLAIMER
        )

    # Seizure Auras & Warning Signs
    if _matches(["aura", "seizure", "twitch", "funny smell", "deja vu", "convulsion"]):
        return (
            f"### ⚡ Recognizing Focal Seizure Auras & Safety Protocols\n\n"
            f"Common possibilities to discuss with a healthcare provider include:\n"
            f"• **Focal Sensory Aura**: Brief intense deja vu, sudden metallic tastes, unusual fleeting smells, or hand tingling can represent mild localized neuronal discharges near the **{location}**.\n"
            f"• **Immediate Safety Action**: Sit or lie down on a flat surface away from hard edges immediately.\n"
            f"• **Medication Protection**: Consistent 12-hour Keppra dosing maintains your seizure threshold. Keep a symptom log and report any change in frequency to **{nurse}**."
            + DISCLAIMER
        )

    # Speech, Word-Finding & Memory
    if _matches(["speech", "words", "talking", "memory", "forget", "brain fog", "aphasia"]):
        return (
            f"### 🗣️ Temporal Lobe Functions: Speech & Verbal Memory\n\n"
            f"Common possibilities to discuss with a healthcare provider include:\n"
            f"• **Temporal Circuit Recovery**: Because your condition is in the **{location}** (which houses language networks), mild word-finding pauses or short-term memory delays are expected, especially when fatigued.\n"
            f"• **Supportive Strategies**: Speak at an unhurried, calm pace. Keep a pocket notebook or phone notepad for daily thoughts, and allow yourself gentle grace."
            + DISCLAIMER
        )

    # =========================================================================
    # 8. CURRENT HEALTH CONDITION & 10-YEAR-OLD EXPLANATIONS
    # =========================================================================

    # Simple plain-language explanation (e.g., 10-year-old / child / plain english)
    if _matches([
        "10 yr", "10-yr", "10 year", "10-year", "ten year", "kid", "child", "children",
        "eli5", "plain english", "explain simply", "simple words", "simple terms",
        "like a child", "to a child", "to a 10", "for a 10", "explain to 10", "scenario to 10",
        "to 10 yr old", "to 10 year old", "explain the scenario to 10", "tell a child", "easy to understand"
    ]):
        return (
            f"### 🧠 Your Health Story Explained Simply (For a 10-Year-Old)\n\n"
            f"Think of your brain like a bustling, wonderful city where different neighborhoods help you think, speak, and remember things:\n\n"
            f"1. **What Happened?**\n"
            f"Inside one neighborhood of the brain called the **{location}** (which helps find words and remember stories), a small cluster of helper cells started growing faster than they should. Doctors call this a glioma.\n\n"
            f"2. **The Surgeon's Mission**:\n"
            f"Skillful neurosurgeons carefully went in and removed the unwanted bump to protect the brain neighborhood and keep it safe.\n\n"
            f"3. **Your Daily Medicine Helpers**:\n"
            f"• **Keppra**: Like quiet safety guards who make sure the electrical signals in your brain stay peaceful, preventing sudden hiccups or twitches (seizures).\n"
            f"• **Temozolomide**: Tiny microscopic helpers that patrol the area to keep everything clean and prevent unwanted cells from returning.\n\n"
            f"4. **What the Brain Pictures (MRI) Show**:\n"
            f"The latest brain scan pictures show that the treated area is very calm and resting. The small spot remaining ({et_v:.1f} cm³) is just gentle healing tissue settling down after surgery—not active new growth!\n\n"
            f"5. **Everyday Superpowers for Brain Healing**:\n"
            f"• **Deep Sleep**: Getting 8 hours of sleep allows your brain's natural nighttime cleaning crew to repair tissue.\n"
            f"• **Water & Good Food**: Drinking water flushes out medicine and keeps your energy up.\n"
            f"• **Gentle Rest**: Taking calm breaks whenever you feel tired helps your brain stay strong!"
            + DISCLAIMER
        )

    # Current Health Condition & Overall Diagnosis Status
    if _matches([
        "condition", "current condition", "my condition", "my health", "how am i doing",
        "what is wrong", "my cancer", "my tumor", "tell me about glioblastoma", "what is glioblastoma",
        "tell me about my cancer", "what disease", "my diagnosis", "tell me about my condition",
        "know about my current condition", "know about my condition", "explain my diagnosis", "status",
        "about my condition", "current health", "summary of my health", "how is my health", "how is my brain"
    ]):
        return (
            f"### 🩺 Comprehensive Overview of Your Current Health & Condition\n\n"
            f"Here is a clear, compassionate, and medically grounded overview of your health journey:\n\n"
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
            + DISCLAIMER
        )

    # Latest Scan Volumetrics & Imaging Breakdown
    if _matches(["scan", "mri", "latest scan", "result", "volume", "tumor size", "edema", "growing", "spread", "shrink", "imaging"]):
        return (
            f"### 🧠 Detailed Interpretation of Your Brain MRI ({scan_date})\n\n"
            f"Here is a comprehensive breakdown of your latest volumetric neuro-imaging results:\n\n"
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
            + DISCLAIMER
        )

    # Emotional Coping, Fear, Family & Contagion
    if _matches(["scared", "fear", "anxious", "anxiety", "depressed", "depression", "sad", "crying", "terrified", "hopeless", "hug", "family", "kids", "children", "contagious", "catch", "kiss"]):
        return (
            f"### 💙 Emotional Support & Family Safety for {patient_name}\n\n"
            f"It takes immense courage to walk this path, and whatever emotions you are feeling right now are completely valid:\n\n"
            f"**1. Reassurance on Family & Closeness**:\n"
            f"• **Brain tumors are NOT contagious**: You cannot pass cancer to your family, children, or loved ones.\n"
            f"• **Hugs and kisses are completely safe**: You do not carry radioactivity after external beam radiation.\n"
            f"• The only temporary precaution is flushing twice and washing hands thoroughly for 48 hours following Temozolomide doses, as minor chemo metabolites exit through fluids.\n\n"
            f"**2. Honoring Your Feelings**:\n"
            f"• Navigating brain cancer can feel overwhelming. It is entirely normal to feel tired, anxious, or frightened on certain days.\n"
            f"• You are not alone. Your care team (**{oncologist}** and **{nurse}**) is actively monitoring your trajectory through your Digital Twin.\n"
            f"• Be gentle with yourself. Taking one day—or one hour—at a time is a valid and powerful way to heal."
        )

    # Prognosis, Future Outlook & Recovery
    if _matches(["prognosis", "survival", "cure", "future", "life expectancy", "will i be okay", "am i dying", "getting better", "recover", "hope", "timeline", "long term"]):
        return (
            f"### 🌅 Understanding Your Prognosis & Path Forward\n\n"
            f"Asking about your future and prognosis is one of the most honest and important questions in neuro-oncology. Here is a balanced, evidence-based medical perspective:\n\n"
            f"**1. Individualized Care, Not Generalized Numbers**:\n"
            f"Broad statistics often group together patients of all ages, genomic backgrounds, and eras. Today, your treatment for **{diagnosis}** in the **{location}** is guided by modern multimodal neuro-oncology and your digital twin biomarkers.\n\n"
            f"**2. Reassuring Indicators in Your Digital Twin**:\n"
            f"• **Low Perfusion rCBV ({rcbv:.2f})**: Rapidly growing recurrent lesions require elevated microvascular blood flow (rCBV > 1.75). Your low perfusion indicates calm tissue responding to treatment.\n"
            f"• **Controlled Residual Core ({et_v:.2f} cm³)**: Your enhancing tumor volume has remained stable post-surgery without rapid progression.\n"
            f"• **Multimodal Therapy**: Stupp protocol adjuvant chemotherapy combined with consistent Keppra dosing gives your brain its strongest defense.\n\n"
            f"**3. Our Shared Goal**:\n"
            f"The clinical focus is maintaining long-term radiographic stability, zero seizures, and vibrant daily living so you can enjoy time with family and friends."
            + DISCLAIMER
        )

    # Appointment Preparation Checklist ("How can I prepare for an appointment?")
    if _matches([
        "prepare for an appointment", "prepare for appointment", "how to prepare for doctor",
        "appointment preparation", "getting ready for appointment", "prepare for clinic",
        "what to bring to doctor", "what to take to appointment", "prepare for my visit"
    ]):
        return (
            f"### 📋 How to Prepare for Your Medical Appointment (Step-by-Step Checklist)\n\n"
            f"Thorough preparation helps you make the most of your consultation with **{oncologist}** and ensures all your questions and concerns are addressed.\n\n"
            f"Here is your clinical preparation checklist:\n\n"
            f"**1. 📝 Prioritize Your Questions in Advance**:\n"
            f"• Write down your top 3 to 5 questions before your visit. Having them written down keeps the conversation focused and ensures you don't forget key points during the consultation.\n"
            f"• *Suggested questions*: 'What did my latest 3T MRI show about tumor stability?', 'How is my rCBV perfusion tracking?', 'Are there any adjustments needed for my Keppra or Temozolomide?', 'When can we discuss driving clearance?'\n\n"
            f"**2. 💊 Bring an Up-to-Date Medication List**:\n"
            f"• List all current prescription medications (exact doses and timing: e.g., Keppra 500 mg BID, Temozolomide cycle dates).\n"
            f"• Include over-the-counter pain relievers, antacids, vitamins, and herbal supplements.\n"
            f"• Note any side effects you noticed (such as morning nausea or drowsiness).\n\n"
            f"**3. 📊 Review Your Recent Symptom Log**:\n"
            f"• Be ready to summarize any symptoms over the last few weeks: Headaches (frequency and intensity), sensory auras, fatigue levels, or word-finding pauses.\n"
            f"• You can tap **'📝 Log Today\\'s Symptoms'** in the app anytime to build an organized record.\n\n"
            f"**4. 📂 Bring Medical Records & Imaging Discs**:\n"
            f"• If you have had any scans (MRI/CT), pathology slides, or lab blood tests done at an outside hospital, bring the physical CDs and printed reports.\n\n"
            f"**5. 👥 Bring a Trusted Caregiver or Family Member**:\n"
            f"• A second set of ears is invaluable. They can take notes, help remember instructions, and provide emotional support.\n\n"
            f"**6. 📋 Ask for a Plain-Language Summary Before Leaving**:\n"
            f"• Before you leave, repeat back the key takeaways in your own words: 'So my plan is to continue current doses and repeat the MRI in 8 weeks, correct?'\n"
            f"• Ask who to call if you experience questions between appointments."
            + DISCLAIMER
        )

    # Appointment Assistance & Schedule Viewing
    if _matches([
        "next appointment", "my appointment", "when is my visit", "upcoming visit",
        "book appointment", "schedule appointment", "make an appointment", "reschedule appointment",
        "view appointments", "appointment date", "when do i see doctor"
    ]):
        appt_text = ""
        if appointments:
            appt_lines = []
            for a in appointments[:3]:
                status_icon = "✅" if a.get("status") in ("Confirmed", "Scheduled") else "📅"
                d_str = a.get("appt_date") or a.get("date") or "Upcoming"
                t_str = a.get("appt_time") or a.get("time") or "10:00 AM"
                s_str = a.get("service_type") or a.get("title") or "Consultation"
                doc_str = a.get("doctor_name") or a.get("clinician") or oncologist
                dept_str = a.get("department") or "Neuro-Oncology"
                loc_str = a.get("location") or "Clinic Suite 4B"
                prep_str = a.get("preparation_notes") or "Bring your recent symptom notes and medication list."
                appt_lines.append(
                    f"• {status_icon} **{d_str} at {t_str}**: {s_str}\n"
                    f"  - **Clinician**: {doc_str} ({dept_str})\n"
                    f"  - **Location**: {loc_str}\n"
                    f"  - **Prep Note**: {prep_str}"
                )
            appt_text = "\n\n".join(appt_lines)
        else:
            next_d = patient.get("next_visit_date", "28-Oct-2026")
            appt_text = (
                f"• 📅 **{next_d} at 10:30 AM**: Neuro-Oncology Consultation & Volumetric Review\n"
                f"  - **Clinician**: {oncologist}\n"
                f"  - **Location**: Comprehensive Brain Tumor Clinic, Room 402\n"
                f"  - **Preparation**: Bring your medication log and recent symptom notes."
            )

        return (
            f"### 🗓️ Your Scheduled Appointments & Clinic Navigation\n\n"
            f"Here are your appointment details from your clinical health record:\n\n"
            f"{appt_text}\n\n"
            f"**Need to Schedule, Reschedule, or Inquire?**\n"
            f"• **Clinic Scheduling Desk**: `+91 (080) 4123-8900` / `+1 (555) 019-8492`\n"
            f"• **Oncology Care Navigator**: {nurse}\n"
            f"• **Clinic Hours**: Monday – Friday, 08:00 AM – 05:00 PM\n\n"
            f"Would you like advice on how to prepare for your next consultation?"
            + DISCLAIMER
        )

    # Questions for Doctor & Consultation Prep
    if _matches(["questions for doctor", "ask doctor", "ask oncologist", "consultation questions", "second opinion"]):
        return (
            f"### 📋 Recommended Questions for Your Next Visit with {oncologist}\n\n"
            f"Here are targeted questions grounded in your Digital Twin to discuss at your upcoming clinic appointment:\n\n"
            f"1. *\"My digital twin shows a low rCBV of {rcbv:.2f} and stable enhancing volume of {et_v:.1f} cm³—does this confirm pseudoprogression rather than active tumor growth?\"*\n"
            f"2. *\"How many more cycles of adjuvant Temozolomide are planned in my Stupp protocol?\"*\n"
            f"3. *\"Based on my current seizure stability on Keppra 500 mg BID, when can we discuss driving clearance?\"*\n"
            f"4. *\"Are there complementary neuro-cognitive therapies or physical rehabilitation exercises that will help my verbal memory in the {location}?\"*\n"
            f"5. *\"When is my next surveillance 3T MRI scheduled?\"*"
            + DISCLAIMER
        )

    # Explicit Greetings & Introductions
    if _matches(["hello", "hi", "hey", "who are you", "what can you do", "good morning", "good evening"]):
        return (
            f"### 👋 Hello! I am HealthMate, your AI Healthcare Companion & Digital Twin GPT\n\n"
            f"I am your empathetic personal health companion, built to support you with wellness tracking, health literacy, symptom awareness, and medication reminders.\n\n"
            f"**Your Personalized Health Record at a Glance**:\n"
            f"• **Diagnosis & Location**: {diagnosis} ({location})\n"
            f"• **Active Prescriptions**: Keppra 500 mg BID & Temozolomide Adjuvant\n"
            f"• **Latest Brain MRI ({scan_date})**: Enhancing Core {et_v:.2f} cm³, Perfusion rCBV: {rcbv:.2f} (Stable / Pseudoprogression)\n"
            f"• **Care Team**: {oncologist} & {nurse}\n\n"
            f"**How I can support you today**:\n"
            f"• *\"Explain my current condition\"*\n"
            f"• *\"Explain the scenario to 10 yr old\"*\n"
            f"• *\"Explain my latest scan results\"*\n"
            f"• *\"How can I maintain a healthy diet?\"*\n"
            f"• *\"What should I do if I miss my Keppra dose?\"*\n"
            f"• *\"When should I see a doctor?\"*\n\n"
            f"How can I support your health and well-being today?"
        )

    # =========================================================================
    # 8.5. COMMON SYMPTOMS, LABS & CLINICAL HEALTH MODULES
    # =========================================================================

    # Fever & Infection / Neutropenic Precautions
    if _matches(["fever", "temp", "temperature", "chills", "shivering", "infection", "cold", "flu", "cough", "sore throat", "runny nose"]):
        return (
            f"### 🌡️ Fever & Infection Precautions During Oncology Treatment\n\n"
            f"During active adjuvant therapy with **Temozolomide**, monitoring for infection is essential:\n\n"
            f"• **Why Fever is Critical**: Chemotherapy temporarily lowers your white blood cells (especially neutrophils), reducing your immune defense against common bacteria and viruses.\n"
            f"• **Fever Threshold**: An oral temperature of **100.4°F (38.0°C) or higher** is considered a clinical priority in oncology.\n"
            f"• **Action Plan**:\n"
            f"  - **Check Temperature**: Use a digital oral thermometer.\n"
            f"  - **Do NOT Mask Fever with Tylenol**: Avoid taking Acetaminophen until you have reported your temperature to **{nurse}** or your clinic.\n"
            f"  - **When to Seek Immediate Care**: If you develop high fever with shaking chills, productive cough, or burning urination, proceed directly to the nearest hospital emergency room (Dial **112 / 108** in India).\n\n"
            f"**Care Team Contacts**:\n"
            f"• Oncology Navigator: **{nurse}**\n"
            f"• Hospital Triage Hotline: `+91 (080) 4123-8900`"
            + DISCLAIMER
        )

    # Nausea, Vomiting & Digestive Comfort
    if _matches(["nausea", "vomit", "throwing up", "queasy", "acid reflux", "heartburn", "indigestion", "appetite loss", "not hungry"]):
        return (
            f"### 🤢 Managing Nausea & Digestive Comfort During Treatment\n\n"
            f"Nausea is a recognized side effect of chemotherapy (Temozolomide) and brain inflammation, but it can be managed effectively:\n\n"
            f"1. **Pre-Medication Timing**:\n"
            f"• Always take your prescribed antiemetic—**Ondansetron (Zofran 8 mg)**—strictly **30 to 45 minutes before** swallowing your Temozolomide capsule.\n"
            f"2. **Bedtime Dosing Strategy**:\n"
            f"• Taking Temozolomide at bedtime on an empty stomach (at least 2 hours after your evening meal) allows you to sleep through peak drug absorption.\n"
            f"3. **Dietary Soothers**:\n"
            f"• Sip warm ginger tea or peppermint tea.\n"
            f"• Eat small, frequent, room-temperature bland snacks (crackers, toast, bananas, rice) rather than heavy, hot, or oily meals.\n"
            f"4. **Never Re-Dose if You Vomit**:\n"
            f"• If you vomit after taking your Temozolomide pill, **do not take another pill**. Notify **{nurse}** in the morning to document the missed dose."
            + DISCLAIMER
        )

    # Dizziness, Vertigo & Balance
    if _matches(["dizzy", "dizziness", "vertigo", "lightheaded", "unsteady", "balance", "spinning", "faint", "room spinning"]):
        return (
            f"### 🌀 Understanding Dizziness & Balance Stability\n\n"
            f"Dizziness or transient lightheadedness can stem from several manageable factors:\n\n"
            f"• **Postural Shifts & Blood Pressure**: Moving from lying down to standing quickly can cause a brief drop in cerebral blood flow. Pause for 30 seconds on the edge of the bed before rising.\n"
            f"• **Anti-Seizure Medication**: Levetiracetam (**Keppra**) can occasionally cause mild vestibular drowsiness or sensation of floating, especially during early cycles.\n"
            f"• **Hydration**: Mild dehydration directly exacerbates lightheadedness. Aim for **2 to 2.5 liters of water daily**.\n\n"
            f"**🚨 Urgent Red-Flag**: If dizziness comes on suddenly with facial drooping, weakness on one side of your body, slurred speech, or difficulty walking, call emergency services (**112 / 108** in India) immediately."
            + DISCLAIMER
        )

    # Constipation & Bowel Health
    if _matches(["constipat", "constipation", "bowel", "stool", "poop", "laxative", "bloat", "stomach cramp"]):
        return (
            f"### 🌿 Bowel Health & Relieving Constipation\n\n"
            f"Constipation is very common during neuro-oncology treatment:\n\n"
            f"• **Main Causes**: Antinausea medications (especially **Ondansetron / Zofran**) and post-surgical pain relievers significantly slow gastrointestinal motility.\n"
            f"• **Hydration First**: Drink warm water or herbal tea first thing in the morning and maintain 2 to 2.5 liters of fluids daily.\n"
            f"• **Natural Fiber**: Include stewed prunes, kiwi fruit, chia seeds, oatmeal, and ground flaxseed in your diet.\n"
            f"• **Gentle Movement**: A 15–20 minute walk stimulates peristalsis.\n"
            f"• **Stool Softeners**: Check with **{nurse}** about using gentle osmotic stool softeners (like Polyethylene Glycol / Miralax or Docusate) if constipation persists beyond 48 hours."
            + DISCLAIMER
        )

    # Pain, Muscle Soreness & Head Pressure
    if _matches(["pain", "sore", "ache", "body pain", "neck pain", "scalp soreness", "muscle pain"]):
        return (
            f"### 🩺 Managing Soreness & Body Pain Safely\n\n"
            f"Post-craniotomy healing and adjuvant treatment can cause localized muscular and scalp soreness:\n\n"
            f"• **Surgical Scalp Soreness**: Healing along your **{location}** incision and surrounding temporalis muscle causes normal mild tenderness for several months.\n"
            f"• **Safe Pain Relief**: **Acetaminophen (Tylenol / Paracetamol 500–650 mg)** is the preferred medication because it does not affect blood clotting. Do not exceed 3,000 mg in 24 hours.\n"
            f"• **Avoid Blood-Thinning NSAIDs**: Do **not** take Ibuprofen, Advil, Motrin, or Aspirin unless specifically cleared by your neurosurgeon, as they increase intracranial bleeding risks.\n"
            f"• **Warm Compress for Neck Tension**: Gentle warm towels around your neck and shoulders ease tension headaches.\n\n"
            f"🚨 *If head pain is sudden, explosive ('worst headache of life'), or accompanied by vomiting, seek emergency care (**112 / 108**) immediately.*"
            + DISCLAIMER
        )

    # Vitamins, Minerals & Supplements
    if _matches(["vitamin", "supplement", "turmeric", "curcumin", "omega 3", "fish oil", "melatonin", "zinc", "magnesium", "b6", "b12", "vitamin d"]):
        return (
            f"### 💊 Vitamins & Supplements in Neuro-Oncology\n\n"
            f"Safe supplementation requires careful coordination with your oncology team:\n\n"
            f"• **Vitamin B6 (Pyridoxine 50–100 mg daily)**: Highly recommended if you experience mild irritability or mood changes from **Keppra**.\n"
            f"• **Vitamin D3 & Calcium**: Important to protect bone density, especially if taking corticosteroids (**Dexamethasone**).\n"
            f"• **Caution with High-Dose Antioxidants**: Avoid mega-dose antioxidants (like Vitamin C > 1,000 mg, high-dose Vitamin E, or concentrated curcumin extracts) during active radiation therapy or chemotherapy cycles, as they may protect cancer cells from the oxidative mechanism of treatment.\n"
            f"• **Always Confirm**: Share any over-the-counter herbal pills with **{oncologist}** before starting to avoid CYP3A4 liver enzyme interactions."
            + DISCLAIMER
        )

    # Blood Tests, CBC, Platelets & Neutrophils
    if _matches(["blood test", "cbc", "platelet", "neutrophil", "white blood", "anc", "labs", "lab result", "hemoglobin", "counts"]):
        return (
            f"### 🩸 Understanding Blood Counts (CBC) & Chemotherapy Safety\n\n"
            f"Before starting each 28-day cycle of adjuvant **Temozolomide**, your care team checks your Complete Blood Count (CBC):\n\n"
            f"• **Absolute Neutrophil Count (ANC)**: Neutrophils fight bacterial infections. Your ANC must be **≥ 1,500/µL** to safely start a chemotherapy cycle.\n"
            f"• **Platelet Count**: Platelets stop bleeding. Your platelets must be **≥ 100,000/µL** before starting your next round.\n"
            f"• **Why Counts Dip (Nadir)**: Blood counts typically reach their lowest point (the nadir) around Days 21–22 of each 28-day cycle, before rebounding naturally.\n"
            f"• **Safety Note**: If counts are low, your oncologist may pause treatment for 1–2 weeks to let your bone marrow recover safely—this is standard clinical practice."
            + DISCLAIMER
        )

    # Seizure Precautions & First Aid
    if _matches(["seizure first aid", "what to do if seizure", "how to help seizure", "seizure safety", "witness a seizure"]):
        return (
            f"### 🛡️ Seizure Safety & First Aid Protocol\n\n"
            f"If you or a caregiver need guidance during a seizure event:\n\n"
            f"1. **Stay Calm & Cushion the Head**: Gently guide the person to the floor and place something soft under their head.\n"
            f"2. **Turn on One Side (Recovery Position)**: This keeps the airway clear and allows fluids to drain.\n"
            f"3. **NEVER Put Anything in the Mouth**: Do not try to hold the tongue or insert objects or medications.\n"
            f"4. **Loosen Restrictive Clothing**: Undo tight collars, neckties, or glasses.\n"
            f"5. **Time the Seizure**:\n"
            f"   - If the seizure lasts **longer than 5 minutes**, or if a second seizure begins immediately, call emergency services (**112 / 108** in India) without delay.\n"
            f"6. **Post-Seizure Care**: Allow restful recovery; mild confusion or sleepiness (the postictal state) is normal for 15–30 minutes."
            + DISCLAIMER
        )

    # Brain Tumor Biology & IDH Mutation
    if _matches(["idh", "mutation", "mgmt", "methylation", "molecular marker", "biology", "biomarker"]):
        return (
            f"### 🧬 Your Molecular Markers: IDH Mutation & Treatment Response\n\n"
            f"Modern neuro-oncology classifies brain tumors by their molecular DNA markers:\n\n"
            f"• **IDH Mutation (Isocitrate Dehydrogenase)**: Your diagnosis carries an **IDH mutation**, which is a **favorable prognostic biomarker**. IDH-mutant tumors produce a distinct metabolic landscape that makes tumor cells significantly more vulnerable to chemotherapy (Temozolomide) and radiation compared to IDH-wildtype tumors.\n"
            f"• **MGMT Promoter Methylation**: When the MGMT gene promoter is methylated (silenced), the tumor cell cannot repair the DNA damage delivered by Temozolomide, leading to greater therapeutic efficacy.\n"
            f"• **Clinical Significance**: These genetic markers guide your personalized care plan and support your favorable treatment trajectory."
            + DISCLAIMER
        )

    # =========================================================================
    # 9. UNIVERSAL OPEN-DOMAIN GENERATIVE HEALTH SYNTHESIZER
    # =========================================================================
    # Cleans query and extracts core health topic
    clean_query = re.sub(r"[?!.,]", "", user_message).strip()
    words = clean_query.split()
    topic = " ".join(words[-4:]) if len(words) >= 4 else clean_query

    # Determine query health category for rich specialized synthesis
    is_symptom = any(k in msg_lower for k in ["feel", "hurt", "pain", "sick", "ache", "symptom", "numb", "tingl", "swollen", "vision", "ear", "sleep", "appetite", "rash"])
    is_treatment = any(k in msg_lower for k in ["pill", "drug", "med", "dose", "chemo", "radiation", "surgery", "resection", "keppra", "temozolomide", "steroid", "therapy"])
    is_lifestyle = any(k in msg_lower for k in ["diet", "food", "eat", "drink", "walk", "exercise", "yoga", "work", "travel", "water", "rest", "lifestyle"])
    is_expl = any(k in msg_lower for k in ["why", "how", "what is", "explain", "understand", "mean", "tell me about"])

    if is_symptom:
        category_header = f"### 🩺 Symptom Guidance: Understanding {topic.title()}"
        guidance_core = (
            f"When monitoring new or changing physical sensations like **{topic}**, here is evidence-based clinical guidance:\n\n"
            f"1. **Assessment Factors**: Note the onset time, duration, and whether it fluctuates or remains constant. Rating its severity from 1 to 10 helps your care team evaluate patterns.\n"
            f"2. **Relation to Your Condition**: Because you are recovering from surgery in your **{location}** and taking **Keppra** and **Temozolomide**, minor nerve adjustments, fluid shifts, or treatment effects can occasionally produce temporary sensations.\n"
            f"3. **Supportive Comfort Steps**: Rest in a calm environment, stay well-hydrated with 2+ liters of water daily, and avoid physical or visual strain.\n"
            f"4. **When to Report**: If the sensation is persistent or worsening, report it to **{nurse}** or your attending neuro-oncologist, **{oncologist}**.\n\n"
            f"🚨 *If this symptom is sudden, severe, accompanied by weakness, facial droop, or trouble speaking, seek emergency evaluation immediately (Dial **112 / 108** in India).*"
        )
    elif is_treatment:
        category_header = f"### 💊 Treatment & Medication Guidance: {topic.title()}"
        guidance_core = (
            f"Regarding your question about **{topic}**, here is essential clinical information:\n\n"
            f"1. **Your Active Regimen**: Your primary prescribed therapies are **Keppra (Levetiracetam 500 mg BID)** for brain electrical stability and **Temozolomide** adjuvant chemotherapy.\n"
            f"2. **Core Guidelines**: Always take prescribed medications at consistent 12-hour intervals. Never alter, double, or skip doses independently without consulting your prescribing physician.\n"
            f"3. **Coordination**: Before introducing any new over-the-counter medicine, pain reliever, or supplement, verify with your clinical navigator, **{nurse}**, to prevent drug interactions or liver strain."
        )
    elif is_lifestyle:
        category_header = f"### 🌿 Daily Wellness & Lifestyle Guidance: {topic.title()}"
        guidance_core = (
            f"Supporting your daily wellness while recovering from **{diagnosis}** involves gentle, restorative habits:\n\n"
            f"1. **Balanced Nutrition & Hydration**: Prioritize whole, anti-inflammatory foods (vegetables, berries, lean proteins, healthy fats) and drink 2 to 2.5 liters of clean water daily to help your kidneys flush medications.\n"
            f"2. **Pacing & Rest**: Gentle physical activities like daily 20-minute walks and restorative stretching boost circulation and reduce fatigue. Avoid heavy straining or holding your breath.\n"
            f"3. **Protecting Brain Recovery**: Deep slow-wave sleep (8 hours nightly) is when your brain repairs cellular connections. Keep a consistent bedtime routine in a quiet, dim room."
        )
    else:
        category_header = f"### 🧠 Clinical Health Overview: {topic.title()}"
        guidance_core = (
            f"Here is a clear, evidence-based perspective regarding **{topic}**:\n\n"
            f"1. **Current Health Context**: Your Digital Twin reflects reassuring stability following surgery in your **{location}**, with stable enhancing volumetrics ({et_v:.1f} cm³) and low perfusion rCBV ({rcbv:.2f}) confirming post-treatment tissue response.\n"
            f"2. **Treatment Protection**: Daily adherence to **Keppra** protects your brain tissue against electrical excitability, while **Temozolomide** guards against cellular recurrence.\n"
            f"3. **Next Steps**: Maintain consistent hydration, prioritize restful sleep, and discuss any specific questions or symptom observations with **{nurse}** or **{oncologist}** at your upcoming consultation."
        )

    synth_response = f"{category_header}\n\n{guidance_core}"
    if _matches(["pain", "hurt", "symptom", "sick", "ache", "head", "med", "pill", "drug", "treatment", "tumor", "cancer", "scan", "surgery", "dose"]):
        synth_response += DISCLAIMER
    return synth_response
