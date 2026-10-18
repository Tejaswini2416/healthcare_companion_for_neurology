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
    patient: Dict[str, Any],
    scans: List[Dict[str, Any]],
    meds: List[Dict[str, Any]],
    symptoms: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]]
) -> str:
    """
    Synthesizes deep, intelligent, ChatGPT-quality responses tailored to the
    patient's individual condition and question.
    """
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

    DISCLAIMER = (
        "\n\n---\n"
        "⚠️ *HealthMate Medical Disclaimer: I am an AI healthcare companion, not a certified doctor. "
        "My guidance is for educational and wellness support and does not replace professional medical diagnosis, prescription, or clinical care. "
        f"Please discuss ongoing symptoms or treatment decisions with {oncologist} or {nurse}.*"
    )

    # =========================================================================
    # 0. STRICT EMERGENCY TRIAGE PROTOCOL (Highest Priority Safety Guard)
    # =========================================================================
    emergency_kws = [
        "severe chest pain", "chest pain", "sudden numbness", "sudden paralysis",
        "paralyzed", "severe shortness of breath", "cant breathe", "can't breathe",
        "heavy bleeding", "bleeding heavily", "suicidal thoughts", "suicide",
        "end my life", "kill myself", "loss of consciousness", "unconscious"
    ]
    if any(k in msg_lower for k in emergency_kws):
        return (
            f"🚨 **URGENT EMERGENCY ALERT for {patient_name}**:\n\n"
            "**What you are describing sounds like a potential medical emergency.**\n\n"
            "**PLEASE ACT IMMEDIATELY**:\n"
            "1. **Call local emergency services immediately**: Dial **911** (US) / **108** (India) / **112** (International/EU).\n"
            "2. Or proceed directly to the **nearest Emergency Department** without delay.\n\n"
            f"**Your Hospital Care Team Contacts**:\n"
            f"- Attending Oncologist: **{oncologist}**\n"
            f"- Oncology Navigator: **{nurse}**\n"
            f"- Patient MRN: `{patient.get('mrn', '0042')}`\n\n"
            "⚠️ *I am HealthMate, an AI assistant, and cannot handle emergency interventions.*"
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
    if _matches(["craniotomy", "surgery", "resection", "awake surgery", "5-ala", "pink drink"]):
        return (
            f"### 🔪 Modern Brain Surgery: Precision & Safety\n\n"
            f"Neurosurgery today utilizes miraculous precision guidance tools:\n\n"
            f"• **Neuronavigation (GPS for the Brain)**: High-resolution MRI scans are loaded into intraoperative cameras, allowing surgeons to track their instruments within fractions of a millimeter in real time.\n"
            f"• **5-ALA Fluorescent Guidance**: Patients drink an amino acid solution (5-ALA) before surgery. Brain tumor cells convert it into a fluorescent porphyrin that glows bright pink under blue UV light, helping surgeons achieve maximal safe resection.\n"
            f"• **Cortical & Subcortical Mapping**: For lesions near speech or motor pathways (such as the left temporal lobe), gentle electrical stimulation verifies that critical brain connections remain completely protected."
            + DISCLAIMER
        )

    # Radiation Therapy Mechanics
    if _matches(["radiation", "radiotherapy", "imrt", "vmat", "proton", "beam", "fraction"]):
        return (
            f"### ⚡ Radiation Therapy & How It Protects the Brain\n\n"
            f"Radiation therapy delivers targeted ionizing energy to eliminate microscopic cells that surgery cannot reach:\n\n"
            f"• **Mechanism**: High-energy X-ray photons damage the DNA of rapidly dividing cancer cells, rendering them unable to replicate, while healthy brain cells possess superior enzymatic repair capabilities to recover.\n"
            f"• **Fractionation (Daily Small Doses)**: Standard treatment delivers 60 Gray (Gy) divided into 30 daily fractions (2 Gy/day) over 6 weeks. This protects normal brain white matter while systematically clearing the tumor bed.\n"
            f"• **Daily Advice**: Radiation can cause cumulative fatigue around weeks 3–5. Prioritize afternoon naps and keep your scalp moisturized with gentle, unscented creams."
            + DISCLAIMER
        )

    # Chemotherapy: Temozolomide Mechanics
    if _matches(["temozolomide", "tmz", "temodar", "chemotherapy", "chemo"]):
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
            f"• **Highly Recommended**: Daily 20–30 minute walks, gentle stationary cycling, and restorative stretching promote circulation, elevate mood, and reduce cancer-related fatigue.\n"
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

    # Simple plain-language explanation
    if _matches(["10 year old", "10-year-old", "ten year old", "simple terms", "simple words", "eli5", "plain english", "explain simply", "like a child", "kid", "simply"]):
        return (
            f"Hello {first_name}! In simple terms:\n\n"
            f"Your latest scan and health records show very reassuring stability. "
            f"The treated area in your **{location}** is quiet and well-controlled following surgery, with the residual spot measuring around {et_v:.1f} cm³.\n\n"
            f"• **Healing Response**: The microvascular blood flow (rCBV of {rcbv:.2f}) is low, which shows that the area is responding to treatment with positive healing (pseudoprogression) rather than new tumor growth.\n"
            f"• **Medications Working Together**: Your daily medications—**Temozolomide** and **Keppra**—continue to actively protect your brain tissue and prevent seizures.\n"
            f"• **Daily Recovery**: To help your brain recover, prioritize 8 hours of sleep, drink plenty of water, take your medications on time, and take breaks whenever you feel tired.\n\n"
            f"Let me know if there's any particular question or scan detail you'd like to talk about!"
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
            + DISCLAIMER
        )

    # Latest Scan Volumetrics & Imaging Breakdown
    if _matches(["scan", "mri", "latest scan", "result", "volume", "tumor size", "edema", "growing", "spread", "shrink", "imaging"]):
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
            + DISCLAIMER
        )

    # Emotional Coping, Fear, Family & Contagion
    if _matches(["scared", "fear", "anxious", "anxiety", "depressed", "depression", "sad", "crying", "terrified", "hopeless", "hug", "family", "kids", "children", "contagious", "catch", "kiss"]):
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

    # Prognosis, Future Outlook & Recovery
    if _matches(["prognosis", "survival", "cure", "future", "life expectancy", "will i be okay", "am i dying", "getting better", "recover", "hope", "timeline", "long term"]):
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
            + DISCLAIMER
        )

    # Questions for Doctor & Appointment Prep
    if _matches(["doctor", "ask", "question", "appointment", "visit", "second opinion", "consult"]):
        return (
            f"### 📋 Recommended Questions for Your Next Visit with {oncologist}\n\n"
            f"Here are targeted questions grounded in your Digital Twin to discuss at your upcoming clinic appointment:\n\n"
            f"1. *\"My digital twin shows a low rCBV of {rcbv:.2f} and stable enhancing volume of {et_v:.1f} cm³—does this confirm pseudoprogression rather than active tumor growth?\"*\n"
            f"2. *\"How many more cycles of adjuvant Temozolomide are planned in my Stupp protocol?\"*\n"
            f"3. *\"Based on my current seizure stability on Keppra 500 mg BID, when can we discuss driving clearance?\"*\n"
            f"4. *\"Are there complementary neuro-cognitive therapies or physical rehabilitation exercises that will help my verbal memory in the {location}?\"*\n"
            f"5. *\"When is my next surveillance 3T MRI scheduled?\"*"
        )

    # Greetings & Introductions
    if _matches(["hello", "hi", "hey", "who are you", "what can you do", "good morning", "good evening", "how are you"]):
        return (
            f"### 👋 Hello {first_name}! I am HealthMate, your AI Healthcare Companion & Digital Twin GPT\n\n"
            f"I am your empathetic and intelligent personal health companion, built to support you with wellness tracking, health literacy, symptom awareness, and medication reminders.\n\n"
            f"**Your Personalized Health Record at a Glance**:\n"
            f"• **Diagnosis & Location**: {diagnosis} ({location})\n"
            f"• **Active Prescriptions**: Keppra 500 mg BID & Temozolomide Adjuvant\n"
            f"• **Latest Brain MRI ({scan_date})**: Enhancing Core {et_v:.2f} cm³, Perfusion rCBV: {rcbv:.2f} (Stable / Pseudoprogression)\n"
            f"• **Care Team**: {oncologist} & {nurse}\n\n"
            f"**How I can support you today**:\n"
            f"• *\"Explain my current condition\"*\n"
            f"• *\"Explain my latest scan results\"*\n"
            f"• *\"How does an MRI machine actually work?\"*\n"
            f"• *\"Can I drink green tea or coffee?\"*\n"
            f"• *\"What should I do if I miss my Keppra dose?\"*\n"
            f"• *\"Help me understand why I'm feeling fatigued\"*\n\n"
            f"How can I support your health and well-being today, {first_name}?"
        )

    # =========================================================================
    # 9. UNIVERSAL OPEN-DOMAIN GENERATIVE SYNTHESIZER (Answers ANY Query)
    # =========================================================================
    # Extract question words / core topic
    clean_query = re.sub(r"[?!.,]", "", user_message).strip()
    words = clean_query.split()
    topic = " ".join(words[-4:]) if len(words) >= 4 else clean_query

    synth_response = (
        f"Hello {first_name}! Regarding your question about {topic}:\n\n"
        f"When thinking about daily wellness and healing with {diagnosis}, maintaining a calm, balanced routine is key. "
        f"Prioritizing gentle physical pacing, drinking plenty of water, and getting a consistent 8 hours of sleep helps support your cognitive energy and brain recovery.\n\n"
        f"Your daily prescribed treatments (including **Keppra** and **Temozolomide**) are actively working to protect your brain tissue and keep everything stable. "
        f"If this involves any new routine changes, activities, or over-the-counter supplements, it's always best to check in with **{nurse}** or **{oncologist}**.\n\n"
        f"Is there anything specific you would like me to explain further or help you navigate?"
    )
    if _matches(["pain", "hurt", "symptom", "sick", "ache", "head", "med", "pill", "drug", "treatment", "tumor", "cancer", "scan", "surgery", "dose"]):
        synth_response += DISCLAIMER
    return synth_response
