"""
Medical Context Generator Module

Generates educational medical context for health screening results.
Non-decisional - provides general information, NOT clinical advice.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from app.core.extraction.base import PhysiologicalSystem
from app.core.inference.risk_engine import RiskLevel
from app.core.llm.gemini_client import GeminiClient, GeminiConfig
from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)


@dataclass
class MedicalContext:
    """Educational medical context for a physiological system."""
    system: PhysiologicalSystem
    
    # Educational content
    system_overview: str = ""
    biomarker_explanations: Dict[str, str] = field(default_factory=dict)
    risk_level_meaning: str = ""
    general_health_info: str = ""
    
    # Lifestyle factors
    lifestyle_factors: List[str] = field(default_factory=list)
    
    # When to seek care
    warning_signs: List[str] = field(default_factory=list)
    
    # Metadata
    is_mock: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system.value,
            "system_overview": self.system_overview,
            "biomarker_explanations": self.biomarker_explanations,
            "risk_level_meaning": self.risk_level_meaning,
            "general_health_info": self.general_health_info,
            "lifestyle_factors": self.lifestyle_factors,
            "warning_signs": self.warning_signs,
            "is_mock": self.is_mock
        }


class MedicalContextGenerator:
    """
    Generates educational medical context using LLM.
    
    NON-DECISIONAL: Provides general health education, not medical advice.
    """
    
    # Pre-defined context for each system (fallback/cache)
    SYSTEM_OVERVIEWS = {
        PhysiologicalSystem.CNS: (
            "The central nervous system (CNS) controls body movement, coordination, "
            "and cognitive functions. Screening indicators include gait patterns, "
            "posture stability, and tremor characteristics."
        ),
        PhysiologicalSystem.CARDIOVASCULAR: (
            "The cardiovascular system circulates blood throughout the body. "
            "Key indicators include heart rate, heart rate variability (HRV), "
            "and blood pressure measurements."
        ),
        PhysiologicalSystem.RENAL: (
            "The renal system filters blood and regulates fluid balance. "
            "Bioimpedance measurements can indicate fluid distribution and "
            "hydration status."
        ),
        PhysiologicalSystem.GASTROINTESTINAL: (
            "The gastrointestinal system processes food and absorbs nutrients. "
            "Abdominal motion patterns and respiratory coupling provide indirect "
            "indicators of GI function."
        ),
        PhysiologicalSystem.SKELETAL: (
            "The skeletal system provides structure and enables movement. "
            "Gait analysis, joint mobility, and posture metrics indicate "
            "musculoskeletal health."
        ),
        PhysiologicalSystem.SKIN: (
            "The skin is the body's largest organ, providing protection and "
            "temperature regulation. Visual analysis can detect texture, "
            "color, and surface characteristics."
        ),
        PhysiologicalSystem.EYES: (
            "The eyes provide vision and reflect overall neurological health. "
            "Blink patterns, gaze stability, and pupil responses are measurable "
            "indicators."
        ),
        PhysiologicalSystem.NASAL: (
            "The nasal and respiratory system enables oxygen exchange. "
            "Breathing patterns, rate, and regularity indicate respiratory health."
        ),
        PhysiologicalSystem.REPRODUCTIVE: (
            "Reproductive health indicators are assessed through autonomic "
            "proxies, as direct non-invasive measurement is limited. "
            "These are indirect indicators only."
        ),
    }
    
    RISK_LEVEL_MEANINGS = {
        RiskLevel.LOW: (
            "A LOW risk level indicates that measured parameters are within "
            "expected normal ranges. Continue current health practices."
        ),
        RiskLevel.MODERATE: (
            "A MODERATE risk level suggests some parameters warrant attention. "
            "Consider discussing with a healthcare provider."
        ),
        RiskLevel.HIGH: (
            "A HIGH risk level indicates parameters that require attention. "
            "Consultation with a healthcare professional is recommended."
        ),
        RiskLevel.ACTION_REQUIRED: (
            "An ACTION_REQUIRED risk level indicates significant abnormalities. "
            "Prompt medical evaluation is strongly recommended."
        ),
    }
    
    SYSTEM_INSTRUCTION = """You are a medical education assistant providing general health information.

CONSTRAINTS:
1. Provide EDUCATIONAL information only, NOT medical advice
2. Do NOT diagnose or suggest specific conditions
3. Do NOT recommend specific treatments
4. Always recommend consulting healthcare professionals
5. Use accessible language for general audiences

Your role is to help people understand what their screening results mean in general terms."""
    
    def __init__(self, client: Optional[GeminiClient] = None):
        """Initialize context generator."""
        if client is None:
            # Use settings for API configuration
            config = GeminiConfig(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model
            )
            self.client = GeminiClient(config)
        else:
            self.client = client
        self._generation_count = 0
        logger.info("MedicalContextGenerator initialized")
    
    def generate_context(
        self,
        system: PhysiologicalSystem,
        risk_level: RiskLevel,
        biomarkers: Optional[List[str]] = None,
        use_llm: bool = True
    ) -> MedicalContext:
        """
        Generate medical context for a physiological system.
        
        Args:
            system: Target physiological system
            risk_level: Current risk level
            biomarkers: Optional list of biomarker names to explain
            use_llm: Whether to use LLM or cached content
            
        Returns:
            MedicalContext with educational information
        """
        context = MedicalContext(system=system)
        
        # Always include cached overviews
        context.system_overview = self.SYSTEM_OVERVIEWS.get(system, "")
        context.risk_level_meaning = self.RISK_LEVEL_MEANINGS.get(risk_level, "")
        
        # Get lifestyle factors and warning signs
        context.lifestyle_factors = self._get_lifestyle_factors(system)
        context.warning_signs = self._get_warning_signs(system, risk_level)
        
        # Use LLM for additional context if available
        if use_llm and self.client.is_available:
            self._enhance_with_llm(context, biomarkers)
        else:
            context.is_mock = True
            context.biomarker_explanations = self._get_biomarker_explanations(system, biomarkers)
        
        self._generation_count += 1
        return context
    
    def _enhance_with_llm(
        self,
        context: MedicalContext,
        biomarkers: Optional[List[str]]
    ) -> None:
        """Enhance context with LLM-generated content."""
        system_name = context.system.value.replace("_", " ").title()
        
        prompt = f"""Provide brief educational information about health screening for the {system_name} system.

Current overview: {context.system_overview}

Please provide:
1. GENERAL HEALTH INFO: 2-3 sentences about maintaining good {system_name.lower()} health
"""
        
        if biomarkers:
            prompt += f"\n2. BIOMARKER EXPLANATIONS: Brief explanation of what each of these biomarkers measures:\n"
            for bm in biomarkers[:5]:  # Limit to 5
                prompt += f"   - {bm}\n"
        
        prompt += """
Keep responses concise and educational. Do not provide medical advice."""
        
        response = self.client.generate(prompt, self.SYSTEM_INSTRUCTION)
        
        if not response.is_mock:
            # Parse general health info
            text = response.text
            if "general health" in text.lower():
                lines = text.split("\n")
                for i, line in enumerate(lines):
                    if "general health" in line.lower() and i + 1 < len(lines):
                        context.general_health_info = lines[i + 1].strip()
                        break
            
            # Parse biomarker explanations
            if biomarkers:
                for bm in biomarkers:
                    if bm.lower() in text.lower():
                        # Find the explanation near the biomarker name
                        idx = text.lower().find(bm.lower())
                        end_idx = text.find("\n", idx + len(bm))
                        if end_idx == -1:
                            end_idx = min(idx + 200, len(text))
                        explanation = text[idx:end_idx].strip()
                        if ":" in explanation:
                            explanation = explanation.split(":", 1)[1].strip()
                        context.biomarker_explanations[bm] = explanation
    
    def _get_lifestyle_factors(self, system: PhysiologicalSystem) -> List[str]:
        """Get general lifestyle factors for a system."""
        factors = {
            PhysiologicalSystem.CNS: [
                "Regular physical activity supports brain health",
                "Quality sleep is essential for cognitive function",
                "Mental stimulation helps maintain neural connections"
            ],
            PhysiologicalSystem.CARDIOVASCULAR: [
                "Regular aerobic exercise strengthens the heart",
                "A balanced diet supports cardiovascular health",
                "Stress management benefits heart function"
            ],
            PhysiologicalSystem.RENAL: [
                "Adequate hydration supports kidney function",
                "Limiting sodium intake helps fluid balance",
                "Regular monitoring if you have risk factors"
            ],
            PhysiologicalSystem.GASTROINTESTINAL: [
                "A fiber-rich diet supports digestive health",
                "Regular meal patterns aid digestion",
                "Staying hydrated helps GI function"
            ],
            PhysiologicalSystem.SKELETAL: [
                "Weight-bearing exercise builds bone strength",
                "Adequate calcium and vitamin D intake",
                "Good posture reduces strain"
            ],
            PhysiologicalSystem.SKIN: [
                "Sun protection prevents skin damage",
                "Hydration and nutrition affect skin health",
                "Regular self-examination for changes"
            ],
            PhysiologicalSystem.EYES: [
                "Regular eye exams are important",
                "Screen breaks reduce eye strain",
                "Protective eyewear when appropriate"
            ],
            PhysiologicalSystem.NASAL: [
                "Avoiding irritants protects airways",
                "Regular exercise supports lung capacity",
                "Good air quality benefits breathing"
            ],
            PhysiologicalSystem.REPRODUCTIVE: [
                "Regular health screenings are important",
                "Stress management affects hormonal balance",
                "Healthy lifestyle supports overall function"
            ],
        }
        return factors.get(system, ["Maintain a healthy lifestyle", "Regular medical checkups"])
    
    def _get_warning_signs(self, system: PhysiologicalSystem, risk_level: RiskLevel) -> List[str]:
        """Get warning signs that warrant medical attention."""
        if risk_level == RiskLevel.LOW:
            return ["Seek care if you experience any concerning symptoms"]
        
        general = ["Consult a healthcare provider if symptoms persist"]
        
        specific = {
            PhysiologicalSystem.CNS: [
                "Sudden balance or coordination problems",
                "Unexplained tremors or weakness",
                "Confusion or cognitive changes"
            ],
            PhysiologicalSystem.CARDIOVASCULAR: [
                "Chest pain or discomfort",
                "Shortness of breath at rest",
                "Irregular heartbeat",
                "Unexplained fatigue"
            ],
            PhysiologicalSystem.RENAL: [
                "Changes in urination patterns",
                "Swelling in legs or ankles",
                "Persistent fatigue"
            ],
        }
        
        return specific.get(system, general)
    
    def _get_biomarker_explanations(
        self,
        system: PhysiologicalSystem,
        biomarkers: Optional[List[str]]
    ) -> Dict[str, str]:
        """Get cached biomarker explanations."""
        explanations = {
            "heart_rate": "Heart rate measures the number of heartbeats per minute",
            "hrv_rmssd": "Heart rate variability indicates the variation between heartbeats",
            "gait_variability": "Gait variability measures consistency in walking patterns",
            "posture_entropy": "Posture entropy reflects the complexity of balance control",
            "respiratory_rate": "Respiratory rate is the number of breaths per minute",
            "blink_rate": "Blink rate indicates eye comfort and neurological function",
            "gait_symmetry_ratio": "Gait symmetry compares left and right step patterns",
        }
        
        if not biomarkers:
            return {}
        
        return {bm: explanations.get(bm, f"{bm} is a measured health indicator") for bm in biomarkers}
