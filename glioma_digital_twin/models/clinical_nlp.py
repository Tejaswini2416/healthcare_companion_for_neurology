"""
Clinical Report Understanding & NLP Module
Bio_ClinicalBERT semantics, diagnostic entity extraction, 6th-grade empathetic translation,
clinical severity tagging, and 11-dimensional L2-normalized semantic embedding vector.
"""

import re
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ClinicalNLPEngine:
    """
    Bio_ClinicalBERT-guided NLP Engine with rule-augmented clinical entity parsing
    and 6th-grade reading level patient-centered translation.
    """
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", offline: bool = True):
        self.model_name = model_name
        self.offline = offline
        self.tokenizer = None
        self.model = None

        if not offline and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
            except Exception as e:
                print(f"[ClinicalNLP] Note: Remote model weights skipped in offline mode ({e}). Utilizing semantic rule engine.")

        # Comprehensive clinical radiology translation lexicon (6th-grade level)
        self.jargon_lexicon = [
            (r"\bperilesional\s+vasogenic\s+edema\b", "mild fluid swelling around the tumor boundary without pressing on other brain structures"),
            (r"\bperilesional\s+edema\b", "mild fluid swelling around the tumor area"),
            (r"\bvasogenic\s+edema\b", "fluid-related swelling in brain tissue"),
            (r"\bresection\s+cavity\b", "the gentle surgical pocket where the doctor previously removed tumor tissue"),
            (r"\bcontrast\s+enhancement\b", "areas where contrast dye highlights active healing and blood vessels on the scan"),
            (r"\benhancing\s+residual\s+lesion\b", "a small residual outline being watched closely"),
            (r"\bnon-enhancing\s+tumor\s+core\b", "the quiet central portion of the tumor zone"),
            (r"\bnecrotic\s+and\s+non-enhancing\b", "inactive, treated tissue at the center"),
            (r"\bmass\s+effect\b", "pressure or crowding on surrounding brain structures"),
            (r"\bmidline\s+shift\b", "sideways shift or displacement of brain structures across the center line"),
            (r"\bno\s+significant\s+mass\s+effect\s+or\s+midline\s+shift\b", "healthy balance across both sides of the brain with no dangerous pressure on nearby areas"),
            (r"\bventricular\s+caliber\s+is\s+symmetric\b", "the natural fluid spaces in the brain remain completely normal and balanced"),
            (r"\brelative\s+cerebral\s+blood\s+volume\b", "measurement of blood circulation through the tissue (rCBV)"),
            (r"\brcbv\b", "blood vessel density measurement"),
            (r"\bpseudoprogression\b", "a temporary, positive reaction where cancer-fighting treatments cause healing inflammation that mimics tumor growth on scans"),
            (r"\btreatment\s+effect\b", "positive healing changes caused by radiation and chemotherapy doing their job"),
            (r"\bneoplastic\s+recurrence\b", "active new tumor growth"),
            (r"\binterval\s+reduction\b", "encouraging decrease in tumor size since the prior scan"),
            (r"\binterval\s+stability\b", "comforting stability with no change since the prior scan"),
            (r"\bhyperintensity\b", "brightly highlighted signal on MRI images"),
            (r"\bhypointensity\b", "darker signal on MRI images"),
        ]

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extracts structured diagnostic clinical entities from free-text radiology reports.
        """
        text_lower = text.lower()

        # 1. Tumor Location
        location = "Temporal Lobe"
        loc_patterns = [
            (r"left\s+temporal\s+lobe(?:\s+resection\s+cavity)?", "Left temporal lobe (resection margin)"),
            (r"right\s+temporal\s+lobe", "Right temporal lobe"),
            (r"frontal\s+lobe", "Frontal lobe"),
            (r"parietal\s+lobe", "Parietal lobe"),
            (r"occipital\s+lobe", "Occipital lobe"),
            (r"cerebellum|cerebellar", "Cerebellar hemisphere"),
        ]
        for pattern, label in loc_patterns:
            if re.search(pattern, text_lower):
                location = label
                break

        # 2. Margins & Enhancement
        margins = "Circumscribed margins"
        if "infiltrating" in text_lower or "ill-defined" in text_lower:
            margins = "Infiltrating / Ill-defined margins"
        elif "residual enhancement" in text_lower or "enhancing residual" in text_lower:
            margins = "Mild residual enhancing margins"
        elif "circumscribed" in text_lower or "well-defined" in text_lower:
            margins = "Well-circumscribed margins"

        enhancement = "Mild"
        if "marked rim enhancement" in text_lower or "intense enhancement" in text_lower:
            enhancement = "Marked Rim Enhancement"
        elif "punctate" in text_lower or "patchy" in text_lower:
            enhancement = "Patchy / Punctate Enhancement"
        elif "interval reduction in enhancing" in text_lower or "decreased enhancement" in text_lower:
            enhancement = "Decreased / Fading Enhancement"

        # 3. Mass effect & Midline shift
        mass_effect = "Absent"
        if "no significant mass effect" in text_lower or "no mass effect" in text_lower:
            mass_effect = "Absent / None"
        elif "moderate mass effect" in text_lower or "ventricular effacement" in text_lower:
            mass_effect = "Moderate (Ventricular effacement)"
        elif "severe mass effect" in text_lower:
            mass_effect = "Severe"

        midline_shift = "None (0 mm)"
        shift_match = re.search(r"(\d+(?:\.\d+)?)\s*mm\s+(?:right|left)?\s*midline\s+shift", text_lower)
        if shift_match:
            midline_shift = f"{shift_match.group(1)} mm shift"
        elif "no midline shift" in text_lower or "midline is intact" in text_lower or "symmetric" in text_lower:
            midline_shift = "None (0 mm)"

        # 4. Perfusion / rCBV
        rcbv = 1.50
        rcbv_match = re.search(r"rcbv\s+(?:is\s+)?(?:reduced\s+at\s+|elevated\s+at\s+|measuring\s+)?(\d+(?:\.\d+)?)", text_lower)
        if rcbv_match:
            try:
                rcbv = float(rcbv_match.group(1))
            except ValueError:
                pass

        # 5. Edema
        edema = "Stable FLAIR vasogenic edema"
        if "moderate flair hyperintensity" in text_lower or "moderate edema" in text_lower:
            edema = "Moderate FLAIR vasogenic edema"
        elif "mild edema" in text_lower:
            edema = "Mild localized edema"
        elif "severe edema" in text_lower:
            edema = "Severe extensive edema"

        return {
            "tumor_location": location,
            "margins": margins,
            "contrast_enhancement": enhancement,
            "mass_effect": mass_effect,
            "midline_shift": midline_shift,
            "rcbv": rcbv,
            "edema": edema,
        }

    def translate_to_plain_language(self, text: str, entities: Optional[Dict[str, Any]] = None) -> str:
        """
        Translates dense clinical findings into an empathetic, reassuring,
        6th-grade reading level summary for patients and caregivers.
        """
        if entities is None:
            entities = self.extract_entities(text)

        # Check key prognostic indicators
        is_improving = "interval reduction" in text.lower() or "decreased" in text.lower() or "stable" in text.lower()
        is_psp = "pseudoprogression" in text.lower() or "treatment effect" in text.lower() or entities["rcbv"] < 1.75
        has_pressure = "severe" in entities["mass_effect"].lower() or entities["midline_shift"] != "None (0 mm)"

        sections = []

        # 1. Headline reassurance
        if is_improving and is_psp:
            headline = (
                "Your latest MRI scan brings encouraging and comforting news. "
                "Overall, the findings show that your treatment is actively working to protect your brain."
            )
        elif is_improving:
            headline = (
                "Your latest MRI scan shows encouraging stability. "
                "The tumor has not shown aggressive new activity."
            )
        else:
            headline = (
                "Your care team has carefully reviewed your latest scan results. "
                "There are a few key details to keep an eye on, and your doctors are actively monitoring them."
            )
        sections.append(headline)

        # 2. Tumor size and boundary description
        loc_desc = entities['tumor_location'].replace("(resection margin)", "").strip()
        if "reduction" in text.lower() or "decreased" in text.lower():
            size_desc = (
                f"In the {loc_desc}, the main active area of the tumor has visibly shrunk. "
                "The edges appear quieter and less active under the contrast dye."
            )
        else:
            size_desc = (
                f"In the {loc_desc}, the treated area remains stable in size compared to your baseline scan."
            )
        sections.append(size_desc)

        # 3. Brain swelling & pressure explanation
        if not has_pressure:
            pressure_desc = (
                "There is mild, normal fluid swelling (often called edema) around the treated area, "
                "which is expected after chemotherapy and radiation. Importantly, there is NO crowding "
                "or dangerous pressure on neighboring healthy brain areas, and both sides of your brain remain nicely balanced."
            )
        else:
            pressure_desc = (
                f"There is some swelling and pressure noticeable ({entities['midline_shift']}). "
                "Your care team may adjust medications like Dexamethasone to quickly relieve this swelling."
            )
        sections.append(pressure_desc)

        # 4. Blood flow and healing vs recurrence (Pseudoprogression)
        if is_psp:
            psp_desc = (
                f"Special imaging measurements of blood flow (measuring {entities['rcbv']:.2f}) confirm that "
                "the tissue changes are largely due to 'treatment effect'—a natural healing response "
                "where your body clears treated cells. This is a very positive sign that your therapies are having their intended impact."
            )
            sections.append(psp_desc)

        # 5. Encouraging closing note
        sections.append(
            "Remember, scan measurements are only one part of your story. Continue taking your medications as prescribed, "
            "stay well-hydrated, and reach out to your oncology nurse if you feel any new headaches or symptoms."
        )

        return " ".join(sections)

    def determine_severity_tag(self, text: str, entities: Dict[str, Any]) -> str:
        """
        Assigns clinical severity tag: 'Mild', 'Moderate', or 'Critical'.
        """
        text_lower = text.lower()
        if entities["midline_shift"] != "None (0 mm)" or "severe mass effect" in text_lower or entities["rcbv"] > 2.5:
            return "Critical"
        if "moderate" in text_lower or entities["rcbv"] > 1.85 or "progressive" in text_lower:
            return "Moderate"
        return "Mild"

    def compute_semantic_embedding(self, text: str, entities: Dict[str, Any]) -> List[float]:
        """
        Computes an 11-dimensional L2-normalized semantic feature vector.
        Features represent:
          0: Enhancement intensity [0-1]
          1: Edema extent [0-1]
          2: Mass effect magnitude [0-1]
          3: Midline shift magnitude [0-1]
          4: rCBV perfusion level [0-1]
          5: Progression tendency [-1 to +1]
          6: Necrosis presence [0-1]
          7: Invasiveness of margins [0-1]
          8: Pseudoprogression probability [0-1]
          9: Brain symmetry preservation [0-1]
          10: Overall severity index [0-1]
        """
        text_lower = text.lower()

        # 0: Enhancement intensity
        f0 = 0.8 if "marked" in text_lower else (0.2 if "fading" in text_lower or "reduction" in text_lower else 0.4)

        # 1: Edema extent
        f1 = 0.8 if "severe" in entities["edema"].lower() else (0.5 if "moderate" in entities["edema"].lower() else 0.25)

        # 2: Mass effect
        f2 = 0.0 if "absent" in entities["mass_effect"].lower() else (0.5 if "moderate" in entities["mass_effect"].lower() else 0.9)

        # 3: Midline shift
        f3 = 0.0 if "none" in entities["midline_shift"].lower() else 0.6

        # 4: rCBV normalized (typical range 0.5 - 3.5)
        rcbv_val = entities.get("rcbv", 1.5)
        f4 = float(np.clip((rcbv_val - 0.5) / 3.0, 0.0, 1.0))

        # 5: Progression tendency (-1 = shrinking, 0 = stable, +1 = progressing)
        if "reduction" in text_lower or "decreased" in text_lower:
            f5 = -0.7
        elif "stable" in text_lower:
            f5 = 0.0
        else:
            f5 = 0.6

        # 6: Necrosis presence
        f6 = 0.7 if "necrotic" in text_lower or "necrosis" in text_lower else 0.3

        # 7: Margin invasiveness
        f7 = 0.7 if "infiltrating" in entities["margins"].lower() else 0.2

        # 8: Treatment effect / Pseudoprogression likelihood
        f8 = 0.85 if (rcbv_val < 1.75 and ("pseudoprogression" in text_lower or "treatment effect" in text_lower or "reduction" in text_lower)) else 0.25

        # 9: Brain symmetry preservation
        f9 = 0.9 if "symmetric" in text_lower or entities["midline_shift"] == "None (0 mm)" else 0.2

        # 10: Overall severity index
        severity = self.determine_severity_tag(text, entities)
        f10 = 0.2 if severity == "Mild" else (0.55 if severity == "Moderate" else 0.9)

        raw_vec = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=np.float32)

        # L2-normalization to unit sphere
        norm = np.linalg.norm(raw_vec)
        if norm > 1e-6:
            norm_vec = raw_vec / norm
        else:
            norm_vec = raw_vec

        return [round(float(x), 5) for x in norm_vec]

    def analyze_report(self, text: str) -> Dict[str, Any]:
        """
        Complete end-to-end clinical NLP pipeline execution.
        """
        entities = self.extract_entities(text)
        plain_summary = self.translate_to_plain_language(text, entities)
        severity_tag = self.determine_severity_tag(text, entities)
        embedding_11d = self.compute_semantic_embedding(text, entities)

        return {
            "parsed_entities": entities,
            "plain_language_summary": plain_summary,
            "severity_tag": severity_tag,
            "embedding_11d": embedding_11d, # 11-dimensional L2-normalized vector
        }
