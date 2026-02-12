"""
Screening Service - Centralized Health Screening Logic
"""
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import HTTPException
from app.core.extraction.base import PhysiologicalSystem, BiomarkerSet, Biomarker
from app.core.inference.risk_engine import RiskEngine, CompositeRiskCalculator, SystemRiskResult, RiskScore
from app.core.llm.multi_llm_interpreter import MultiLLMInterpreter

logger = logging.getLogger(__name__)

class ScreeningService:
    """
    Service class to handle the health screening business logic.
    Decouples the logic from FastAPI endpoints and hardware manager.
    """
    
    def __init__(self, risk_engine: Optional[RiskEngine] = None, interpreter: Optional[MultiLLMInterpreter] = None):
        self.risk_engine = risk_engine or RiskEngine()
        self.interpreter = interpreter or MultiLLMInterpreter()
        self.composite_calc = CompositeRiskCalculator()

    async def process_screening(
        self, 
        patient_id: str, 
        systems_input: List[Dict[str, Any]], 
        include_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Processes screening data and returns risk assessments.
        
        Args:
            patient_id: ID of the patient.
            systems_input: List of system data (name and biomarkers).
            include_validation: Whether to run LLM validation.
            
        Returns:
            A dictionary containing the screening results.
        """
        screening_id = f"SCR-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now()
        
        from app.core.inference.risk_engine import TrustedRiskResult
        
        trusted_results: Dict[PhysiologicalSystem, TrustedRiskResult] = {}
        system_results: Dict[PhysiologicalSystem, SystemRiskResult] = {}
        response_results: List[Dict[str, Any]] = []
        
        for sys_input in systems_input:
            system_name = sys_input.get("system", "")
            biomarkers_input = sys_input.get("biomarkers", [])
            
            logger.info(f"Processing system: {system_name} with {len(biomarkers_input)} biomarkers")
            
            # Parse system
            system = self._parse_system(system_name)
            
            # Convert biomarker inputs
            biomarkers = [
                Biomarker(
                    name=bm.get("name"),
                    value=bm.get("value"),
                    unit=bm.get("unit") or "",
                    confidence=1.0,
                    normal_range=tuple(bm.get("normal_range")) if bm.get("normal_range") and len(bm.get("normal_range")) == 2 else None
                )
                for bm in biomarkers_input
            ]
            
            # Create BiomarkerSet
            biomarker_set = BiomarkerSet(system=system, biomarkers=biomarkers)
            
            # Run risk calculation
            trusted_result = self.risk_engine.compute_risk_with_validation(biomarker_set, plausibility=None)
            trusted_results[system] = trusted_result
            
            # Build response item
            if trusted_result.was_rejected:
                response_results.append({
                    "system": system.value,
                    "risk_level": "unknown",
                    "risk_score": 0.0,
                    "confidence": 0.0,
                    "alerts": [trusted_result.rejection_reason] if trusted_result.rejection_reason else [],
                    "is_trusted": False,
                    "was_rejected": True,
                    "caveats": trusted_result.caveats
                })
            elif trusted_result.risk_result is not None:
                result = trusted_result.risk_result
                system_results[system] = result
                
                response_results.append({
                    "system": system.value,
                    "risk_level": result.overall_risk.level.value,
                    "risk_score": round(result.overall_risk.score, 1),
                    "confidence": round(trusted_result.trust_adjusted_confidence, 2),
                    "alerts": result.alerts,
                    "is_trusted": trusted_result.is_trusted,
                    "was_rejected": False,
                    "caveats": trusted_result.caveats
                })

        # Calculate composite risk
        composite, rejected_systems = self.composite_calc.compute_composite_risk_from_trusted(trusted_results)
        
        # Validation (LLM)
        validation_status = None
        requires_review = False
        
        if include_validation and system_results:
            try:
                # Note: MultiLLMInterpreter is being converted to async
                interpretation = await self.interpreter.interpret_composite_risk(
                    system_results=system_results,
                    composite_risk=composite,
                    trust_envelope=None
                )
                validation_status = "validated" if interpretation.validation_passed else "needs_review"
                requires_review = not interpretation.validation_passed or len(rejected_systems) > 0
                logger.info(f"MultiLLM validation: mode={interpretation.pipeline_mode}, passed={interpretation.validation_passed}")
            except Exception as e:
                logger.warning(f"MultiLLM validation failed: {e}")
                validation_status = "plausible" if not rejected_systems else "partial"
                requires_review = len(rejected_systems) > 0
        
        return {
            "screening_id": screening_id,
            "patient_id": patient_id,
            "timestamp": timestamp,
            "overall_risk_level": composite.level.value,
            "overall_risk_score": round(composite.score, 1),
            "overall_confidence": round(composite.confidence, 2),
            "system_results": response_results,
            "system_results_internal": system_results, # Internal storage
            "trusted_results": trusted_results,
            "composite_risk": composite,
            "rejected_systems": rejected_systems,
            "validation_status": validation_status,
            "requires_review": requires_review
        }

    def _parse_system(self, system_name: str) -> PhysiologicalSystem:
        """Helper to parse system name string to PhysiologicalSystem enum."""
        try:
            return PhysiologicalSystem.from_string(system_name)
        except ValueError:
            logger.warning(f"Unknown system name: {system_name}, defaulting to CNS")
            return PhysiologicalSystem.CNS
