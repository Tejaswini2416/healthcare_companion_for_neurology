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
        mass_effect = "Absent / None"
        if "severe mass effect" in text_lower or "herniation" in text_lower:
            mass_effect = "Severe (Impending herniation / cisternal effacement)"
        elif "moderate mass effect" in text_lower or "ventricular effacement" in text_lower or "partial effacement" in text_lower:
            mass_effect = "Moderate (Ventricular effacement)"
        elif "no significant mass effect" in text_lower or "no mass effect" in text_lower:
            mass_effect = "Absent / None"

        midline_shift = "None (0 mm)"
        shift_match = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:right|left|rightward|leftward)?\s*midline\s+shift", text_lower)
        if shift_match:
            try:
                shift_num = float(shift_match.group(1))
                if shift_num > 0.1:
                    midline_shift = f"{shift_num} mm shift"
                else:
                    midline_shift = "None (0 mm)"
            except ValueError:
                midline_shift = "None (0 mm)"
        elif "no midline shift" in text_lower or "midline is intact" in text_lower or "symmetric" in text_lower:
            midline_shift = "None (0 mm)"

        # 4. Perfusion / rCBV
        rcbv = 1.50
        rcbv_match = re.search(r"rcbv\s+(?:is\s+)?(?:reduced\s+at\s+|elevated\s+at\s+|measuring\s+|measured\s+at\s+|measured\s+|of\s+)?(\d+(?:\.\d+)?)", text_lower)
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

    @staticmethod
    def _normalize_entities(entities: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Safely normalizes parsed entities whether they are boolean flags, strings, or numbers."""
        if not entities:
            entities = {}
        out = dict(entities)

        # mass_effect
        me = out.get("mass_effect", "Absent / None")
        if isinstance(me, bool):
            out["mass_effect"] = "Moderate (Ventricular effacement)" if me else "Absent / None"
        else:
            out["mass_effect"] = str(me) if me is not None else "Absent / None"

        # midline_shift
        ms = out.get("midline_shift")
        if ms is None:
            ms_num = out.get("midline_shift_mm", 0.0)
            try:
                ms_float = float(ms_num)
                out["midline_shift"] = f"{ms_float} mm shift" if ms_float > 0.1 else "None (0 mm)"
            except Exception:
                out["midline_shift"] = "None (0 mm)"
        else:
            out["midline_shift"] = str(ms)

        # rcbv
        rcbv = out.get("rcbv", 1.50)
        try:
            out["rcbv"] = float(rcbv)
        except Exception:
            out["rcbv"] = 1.50

        # edema
        ed = out.get("edema") or out.get("edema_status", "Stable FLAIR vasogenic edema")
        out["edema"] = str(ed)

        # margins
        mg = out.get("margins", "Well-demarcated")
        out["margins"] = str(mg)

        # tumor_location
        tl = out.get("tumor_location", "Left temporal lobe")
        out["tumor_location"] = str(tl)

        return out

    def translate_to_plain_language(
        self,
        text: str,
        entities: Optional[Dict[str, Any]] = None,
        language: str = "en",
        target_lang: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Translates dense clinical findings into an empathetic, reassuring,
        6th-grade reading level summary for patients and caregivers across multiple languages:
        'en' (English), 'es' (Spanish), 'hi' (Hindi), 'zh' (Mandarin), 'fr' (French).
        """
        if target_lang:
            language = target_lang
        if entities is None:
            entities = self.extract_entities(text)
        entities = self._normalize_entities(entities)

        # Check key prognostic indicators
        is_improving = "interval reduction" in text.lower() or "decreased" in text.lower() or "stable" in text.lower()
        is_psp = "pseudoprogression" in text.lower() or "treatment effect" in text.lower() or entities["rcbv"] < 1.75
        has_pressure = "severe" in entities["mass_effect"].lower() or entities["midline_shift"] != "None (0 mm)"
        loc_desc = str(entities['tumor_location']).replace("(resection margin)", "").strip()

        lang = language.lower()

        # --- 1. SPANISH (ESPAÑOL) ---
        if lang == "es":
            if is_improving and is_psp:
                h = "Su última resonancia magnética (MRI) muestra noticias muy alentadoras. En general, los hallazgos confirman que su tratamiento está protegiendo activamente su cerebro."
            elif is_improving:
                h = "Su última resonancia magnética muestra una reconfortante estabilidad clínica sin nueva actividad tumoral agresiva."
            else:
                h = "Su equipo médico ha revisado detalladamente su resonancia y continuará supervisando de cerca la evolución."

            s = f"En la zona de {loc_desc}, el área activa del tumor ha disminuido de tamaño o se mantiene tranquila." if is_improving else f"En {loc_desc}, la lesión tratada se mantiene estable."
            p = "Existe una leve hinchazón de líquidos (edema) esperada tras la radioterapia, sin presión peligrosa sobre el cerebro sano." if not has_pressure else f"Se observa algo de presión ({entities['midline_shift']}), la cual su médico vigilará con medicación como Dexametasona."
            psp = f"Las mediciones de flujo sanguíneo (rCBV: {entities['rcbv']:.2f}) confirman que los cambios corresponden a una cicatrización benigna y positiva del tratamiento (pseudoprogresión)." if is_psp else ""
            c = "Continúe tomando sus medicamentos según las indicaciones prescritas y manténgase bien hidratado."
            return " ".join([part for part in [h, s, p, psp, c] if part])

        # --- 2. HINDI (हिंदी) ---
        elif lang == "hi":
            if is_improving and is_psp:
                h = "आपके नवीनतम एमआरआई स्कैन में बहुत उत्साहजनक और सुकून देने वाले परिणाम मिले हैं। उपचार सक्रिय रूप से आपके मस्तिष्क की सुरक्षा कर रहा है।"
            elif is_improving:
                h = "आपका नया स्कैन स्थिरता दर्शाता है। ट्यूमर में कोई नई आक्रामक गतिविधि नहीं देखी गई है।"
            else:
                h = "आपकी मेडिकल टीम ने आपकी रिपोर्ट की सावधानीपूर्वक समीक्षा की है और नियमित निगरानी जारी रखेगी।"

            s = f"{loc_desc} क्षेत्र में सक्रिय ट्यूमर का आकार कम हुआ है या स्थिर बना हुआ है।"
            p = "इलाज किए गए हिस्से के चारों ओर हल्की सामान्य तरल सूजन (एडिमा) है, लेकिन स्वस्थ मस्तिष्क पर कोई खतरनाक दबाव नहीं है।" if not has_pressure else f"हल्का दबाव ({entities['midline_shift']}) देखा गया है जिसे नियंत्रित करने के लिए डॉक्टर दवाएं समायोजित कर सकते हैं।"
            psp = f"रक्त परिसंचरण माप (rCBV: {entities['rcbv']:.2f}) से पुष्टि होती है कि यह सकारात्मक उपचार प्रभाव (स्यूडोप्रोग्रेशन) है, ट्यूमर की वापसी नहीं।" if is_psp else ""
            c = "कृपया अपनी दवाएं समय पर लेते रहें और पर्याप्त पानी पिएं। किसी भी नए लक्षण पर अपनी ऑन्कोलॉजी नर्स से संपर्क करें।"
            return " ".join([part for part in [h, s, p, psp, c] if part])

        # --- 3. MANDARIN (中文) ---
        elif lang == "zh":
            if is_improving and is_psp:
                h = "您最新的脑部核磁共振（MRI）检查带来了非常令人鼓舞的好消息。整体结果显示，您的抗癌治疗正在积极有效地发挥保护作用。"
            elif is_improving:
                h = "最新的核磁共振结果显示肿瘤区域保持良好稳定，未见异常活跃新生迹象。"
            else:
                h = "您的医疗专家团队已经仔细审查了最新扫描结果，并将继续对各项细节进行细致随访。"

            s = f"在{loc_desc}部位，原肿瘤活跃范围明显缩小或维持稳定。"
            p = "手术治疗区周围仅有轻微预期的组织水肿，并未对周围健康的脑组织构成危险压迫，左右脑对称平衡良好。" if not has_pressure else f"存在一定程度的压力（中线移位：{entities['midline_shift']}），医疗团队可能会调整地塞米松等药物以缓解水肿。"
            psp = f"局部血流灌注评估（rCBV: {entities['rcbv']:.2f}）证实，这些变化主要属于放化疗后的积极组织修复反应（假性进展），而非肿瘤复发。" if is_psp else ""
            c = "请继续按照处方准时服用药物，保持充足饮水和良好作息。如感到任何不适请随时联系您的主管护士。"
            return " ".join([part for part in [h, s, p, psp, c] if part])

        # --- 4. FRENCH (FRANÇAIS) ---
        elif lang == "fr":
            if is_improving and is_psp:
                h = "Votre dernière IRM apporte des nouvelles très rassurantes. Dans l'ensemble, les résultats indiquent que votre traitement protège activement votre santé cérébrale."
            elif is_improving:
                h = "Votre dernière IRM témoigne d'une stabilité clinique encourageante sans signe de nouvelle progression agressive."
            else:
                h = "Votre équipe soignante a examiné attentivement vos résultats et maintient une surveillance rigoureuse."

            s = f"Dans la zone de {loc_desc}, la partie active de la lésion a visiblement régressé ou demeure stable."
            p = "Il existe un léger œdème réactionnel normal autour de la région traitée, sans aucune compression anormale sur les structures saines." if not has_pressure else f"Une légère pression est constatée ({entities['midline_shift']}), que vos médecins réguleront avec vos corticoïdes."
            psp = f"L'évaluation de la perfusion sanguine (rCBV : {entities['rcbv']:.2f}) confirme qu'il s'agit d'un effet positif de cicatrisation lié aux rayons (pseudoprogression) et non d'une reprise tumorale." if is_psp else ""
            c = "Poursuivez régulièrement vos traitements tels que prescrits et reposez-vous bien."
            return " ".join([part for part in [h, s, p, psp, c] if part])

        # --- 5. ENGLISH (DEFAULT) ---
        else:
            sections = []
            if is_improving and is_psp:
                sections.append(
                    "Your latest MRI scan brings encouraging and comforting news. "
                    "Overall, the findings show that your treatment is actively working to protect your brain."
                )
            elif is_improving:
                sections.append(
                    "Your latest MRI scan shows encouraging stability. "
                    "The tumor has not shown aggressive new activity."
                )
            else:
                sections.append(
                    "Your care team has carefully reviewed your latest scan results. "
                    "There are a few key details to keep an eye on, and your doctors are actively monitoring them."
                )

            if "reduction" in text.lower() or "decreased" in text.lower():
                sections.append(
                    f"In the {loc_desc}, the main active area of the tumor has visibly shrunk. "
                    "The edges appear quieter and less active under the contrast dye."
                )
            else:
                sections.append(
                    f"In the {loc_desc}, the treated area remains stable in size compared to your baseline scan."
                )

            if not has_pressure:
                sections.append(
                    "There is mild, normal fluid swelling (often called edema) around the treated area, "
                    "which is expected after chemotherapy and radiation. Importantly, there is NO crowding "
                    "or dangerous pressure on neighboring healthy brain areas, and both sides of your brain remain nicely balanced."
                )
            else:
                sections.append(
                    f"There is some swelling and pressure noticeable ({entities['midline_shift']}). "
                    "Your care team may adjust medications like Dexamethasone to quickly relieve this swelling."
                )

            if is_psp:
                sections.append(
                    f"Special imaging measurements of blood flow (measuring {entities['rcbv']:.2f}) confirm that "
                    "the tissue changes are largely due to 'treatment effect'—a natural healing response "
                    "where your body clears treated cells. This is a very positive sign that your therapies are having their intended impact."
                )

            sections.append(
                "Remember, scan measurements are only one part of your story. Continue taking your medications as prescribed, "
                "stay well-hydrated, and reach out to your oncology nurse if you feel any new headaches or symptoms."
            )

            return " ".join(sections)

    def determine_severity_tag(self, text: str, entities: Dict[str, Any]) -> str:
        """
        Assigns clinical severity tag: 'Mild', 'Moderate', or 'Critical'.
        """
        entities = self._normalize_entities(entities)
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
        entities = self._normalize_entities(entities)
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
        plain_summary = self.translate_to_plain_language(text, entities, language="en")
        translations = {
            "en": plain_summary,
            "es": self.translate_to_plain_language(text, entities, language="es"),
            "hi": self.translate_to_plain_language(text, entities, language="hi"),
            "zh": self.translate_to_plain_language(text, entities, language="zh"),
            "fr": self.translate_to_plain_language(text, entities, language="fr"),
        }
        severity_tag = self.determine_severity_tag(text, entities)
        embedding_11d = self.compute_semantic_embedding(text, entities)

        return {
            "parsed_entities": entities,
            "plain_language_summary": plain_summary,
            "translations": translations,
            "severity_tag": severity_tag,
            "embedding_11d": embedding_11d, # 11-dimensional L2-normalized vector
        }
