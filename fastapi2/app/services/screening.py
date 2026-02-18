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
from app.core.validation.signal_quality import SignalQualityAssessor, Modality
from app.core.validation.biomarker_plausibility import BiomarkerPlausibilityValidator, PatientContext
from app.core.validation.cross_system_consistency import CrossSystemConsistencyChecker
from app.core.validation.trust_envelope import TrustEnvelopeAggregator, TrustEnvelope
from app.config import settings

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
        self.quality_assessor = SignalQualityAssessor()
        
        # Validation Modules (Initialized based on config)
        self.plausibility_validator = BiomarkerPlausibilityValidator()
        self.consistency_checker = CrossSystemConsistencyChecker()
        self.trust_aggregator = TrustEnvelopeAggregator()
        
        # Load capability flags from central settings
        self._enable_validation = settings.enable_validation
        self._enable_plausibility = settings.enable_plausibility
        self._enable_consistency = settings.enable_consistency
        self._enable_trust_envelope = settings.enable_trust_envelope

    async def process_screening(
        self,
        patient_id: str,
        systems_input: List[Dict[str, Any]],
        include_validation: bool = True,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Processes screening data and returns risk assessments.

        Args:
            patient_id:       ID of the patient.
            systems_input:    List of system data (name and biomarkers).
            include_validation: Whether to run LLM validation.
            patient_context:  Optional dict with age/gender/activity_mode for dynamic
                              plausibility validation. Falls back to static defaults if None.

        Returns:
            A dictionary containing the screening results.
        """
        # Build PatientContext for dynamic plausibility limits
        ctx: Optional[PatientContext] = None
        if patient_context and isinstance(patient_context, dict):
            ctx = PatientContext(
                age=patient_context.get("age", 30),
                gender=patient_context.get("gender", "male"),
                activity_mode=patient_context.get("activity_mode", "resting"),
            )
            logger.info(
                f"PatientContext received: age={ctx.age}, gender={ctx.gender}, "
                f"activity_mode={ctx.activity_mode}"
            )
        screening_id = f"SCR-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now()
        
        from app.core.inference.risk_engine import TrustedRiskResult
        
        trusted_results: Dict[PhysiologicalSystem, TrustedRiskResult] = {}
        system_results: Dict[PhysiologicalSystem, SystemRiskResult] = {}
        plausibility_results: Dict[PhysiologicalSystem, Any] = {}
        biomarker_sets: Dict[PhysiologicalSystem, BiomarkerSet] = {}
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
                    confidence=bm.get("confidence", 1.0),  # ← Use input confidence or default to 1.0,
                    normal_range=tuple(bm.get("normal_range")) if bm.get("normal_range") and len(bm.get("normal_range")) == 2 else None
                )
                for bm in biomarkers_input
            ]
            
            # Create BiomarkerSet
            biomarker_set = BiomarkerSet(system=system, biomarkers=biomarkers)
            biomarker_sets[system] = biomarker_set
            
            # Run plausibility validation
            plausibility_result = None
            if self._enable_validation and self._enable_plausibility:
                try:
                    plausibility_result = self.plausibility_validator.validate(
                        biomarker_set, context=ctx
                    )
                    plausibility_results[system] = plausibility_result
                    logger.info(f"[{system.value}] Plausibility: valid={plausibility_result.is_valid}, score={plausibility_result.overall_plausibility:.2f}")
                except Exception as e:
                    logger.error(f"Plausibility validation failed for {system.value}: {e}", exc_info=True)
            
            # Run risk calculation
            trusted_result = self.risk_engine.compute_risk_with_validation(biomarker_set, plausibility=plausibility_result)
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
        
        # Cross-System Consistency Check
        consistency_result = None
        if self._enable_validation and self._enable_consistency and len(biomarker_sets) > 1:
            try:
                consistency_result = self.consistency_checker.check_consistency(biomarker_sets, system_results)
                logger.info(f"Cross-system consistency: {consistency_result.overall_consistency:.2f}")
            except Exception as e:
                logger.error(f"Consistency check failed: {e}", exc_info=True)
        
        # Trust Envelope Aggregation
        trust_envelope = None
        if self._enable_validation and self._enable_trust_envelope:
            try:
                # We need to assess quality first if we want it in the envelope
                # But assess_data_quality takes raw 'data' which might not be here.
                # For now, we'll use a placeholder or calculate it from confidences if needed.
                # However, the aggregator handles None signal_quality gracefully.
                trust_envelope = self.trust_aggregator.aggregate(
                    signal_quality=None, # Raw data quality not easily accessible here without 'data'
                    plausibility_results=plausibility_results,
                    consistency_result=consistency_result
                )
                logger.info(f"Trust Envelope reliability: {trust_envelope.overall_reliability:.2f}")
            except Exception as e:
                logger.error(f"Trust aggregation failed: {e}", exc_info=True)
        
        # Enforce minimum reliability threshold
        if trust_envelope and trust_envelope.overall_reliability < settings.min_trust_reliability:
            logger.warning(f"Screening reliability ({trust_envelope.overall_reliability:.2f}) is below threshold ({settings.min_trust_reliability:.2f})")
            validation_status = "low_reliability"
            requires_review = True
        
        # Validation (LLM)
        validation_status = None
        requires_review = False
        
        if include_validation and system_results:
            try:
                # Note: MultiLLMInterpreter is being converted to async
                interpretation = await self.interpreter.interpret_composite_risk(
                    system_results=system_results,
                    composite_risk=composite,
                    trust_envelope=trust_envelope
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
            "trust_metadata": trust_envelope.to_dict() if trust_envelope else None,
            "validation_status": validation_status,
            "requires_review": requires_review
        }

    async def assess_data_quality(self, data: Optional[Dict[str, Any]], systems_input: List[Dict[str, Any]]) -> float:
        """
        Assess overall data quality from raw sensor data or biomarker confidence.
        Returns a score from 0 to 1.
        """
        logger.info(f"assess_data_quality called: data={data is not None}, systems_input={len(systems_input) if systems_input else 0}")
        
        # Case 1: We have raw sensor metadata
        if data:
            try:
                # Modality map for assessor
                quality_scores = {}
                
                # Check for camera/frames
                if "camera" in data:
                    quality_scores[Modality.CAMERA] = self.quality_assessor.assess_camera(
                        data["camera"].get("frames", []), 
                        data["camera"].get("timestamps")
                    )
                
                # Check for radar
                if "radar" in data:
                    quality_scores[Modality.RADAR] = self.quality_assessor.assess_radar(data["radar"])
                
                # Check for thermal
                if "thermal" in data:
                    quality_scores[Modality.THERMAL] = self.quality_assessor.assess_thermal(data["thermal"])
                
                if quality_scores:
                    overall = sum(s.overall_quality for s in quality_scores.values()) / len(quality_scores)
                    logger.info(f"Data quality assessment (Raw): {overall:.2f}")
                    return overall
            except Exception as e:
                logger.warning(f"Raw data quality assessment failed: {e}")

        # Case 2: Fallback to checking biomarker confidence if raw data is missing
        if systems_input:
            confidences = []
            for sys in systems_input:
                for bm in sys.get("biomarkers", []):
                    conf = bm.get("confidence")
                    logger.debug(f"Biomarker {bm.get('name')}: confidence={conf}")
                    if conf is not None:
                        confidences.append(float(conf))
            
            logger.info(f"Found {len(confidences)} confidence values")
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                logger.info(f"Data quality assessment (Biomarker Fallback): {avg_conf:.2f}")
                return avg_conf

        logger.warning("No data available for quality assessment - defaulting to low quality (0.0)")
        return 0.0

    def _parse_system(self, system_name: str) -> PhysiologicalSystem:
        """Helper to parse system name string to PhysiologicalSystem enum."""
        try:
            return PhysiologicalSystem.from_string(system_name)
        except ValueError:
            logger.warning(f"Unknown system name: {system_name}, defaulting to CNS")
            return PhysiologicalSystem.CNS
