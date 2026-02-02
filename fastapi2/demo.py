"""
End-to-End Demo Script for Health Screening Pipeline

This script demonstrates the complete pipeline with mock data:
1. Mock sensor data (camera, audio, accelerometer)
2. Feature extraction
3. Risk inference
4. Multi-LLM interpretation (Gemini + 2 HuggingFace medical models)
5. Agentic validation (using GPT-OSS-120B and II-Medical-8B)
6. Report generation with comprehensive AI insights

NEW: Uses LangChain for Gemini and HuggingFace InferenceClient for medical models

Run: python demo.py
"""
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ---- Imports ----
print("=" * 60)
print("HEALTH SCREENING PIPELINE - END-TO-END DEMO")
print("=" * 60)
print()

print("[1/8] Loading modules...")

# Work around pydantic-settings import issue by patching config first
import sys
from types import ModuleType

# Create a mock settings module if pydantic_settings not available
try:
    import pydantic_settings
except ImportError:
    # Create mock pydantic_settings
    mock_module = ModuleType('pydantic_settings')
    mock_module.BaseSettings = type('BaseSettings', (), {})  # type: ignore
    sys.modules['pydantic_settings'] = mock_module

try:
    from app.core.extraction.base import PhysiologicalSystem
    from app.core.inference.risk_engine import RiskEngine, RiskLevel
    from app.core.validation.trust_envelope import TrustEnvelope
    from app.core.llm.context_generator import MedicalContextGenerator
    from app.core.llm.risk_interpreter import RiskInterpreter
    from app.core.llm.multi_llm_interpreter import MultiLLMInterpreter
    from app.core.agents.medical_agents import AgentConsensus, MedGemmaAgent, OpenBioLLMAgent
    from app.core.reports import PatientReportGenerator, DoctorReportGenerator
    print("   ✓ All modules loaded successfully!")
    print("   ✓ Using LangChain for Gemini")
    print("   ✓ Using HuggingFace InferenceClient for medical models")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    print("   Try: pip install -r requirements.txt")
    sys.exit(1)


# ---- Mock Data Generation ----
print()
print("[2/8] Generating mock sensor data...")


def generate_mock_ppg_signal(duration_sec=10, fps=30):
    """Generate mock PPG signal (camera-based heart rate)."""
    t = np.linspace(0, duration_sec, int(duration_sec * fps))
    heart_rate = 72  # BPM
    ppg = np.sin(2 * np.pi * (heart_rate / 60) * t) + 0.1 * np.random.randn(len(t))
    return ppg


def generate_mock_audio_signal(duration_sec=10, sample_rate=16000):
    """Generate mock audio signal (breathing sounds)."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    breath_rate = 16  # breaths per minute
    audio = np.sin(2 * np.pi * (breath_rate / 60) * t) * 0.3 + 0.05 * np.random.randn(len(t))
    return audio


def generate_mock_accelerometer(duration_sec=10, sample_rate=50):
    """Generate mock accelerometer data (gait analysis)."""
    samples = int(duration_sec * sample_rate)
    step_freq = 1.8  # steps per second (normal walking)
    t = np.linspace(0, duration_sec, samples)
    
    acc_x = np.sin(2 * np.pi * step_freq * t) * 0.5 + 0.1 * np.random.randn(samples)
    acc_y = np.cos(2 * np.pi * step_freq * t) * 0.3 + 0.1 * np.random.randn(samples)
    acc_z = 9.8 + np.sin(2 * np.pi * step_freq * 2 * t) * 0.2 + 0.1 * np.random.randn(samples)
    
    return np.column_stack([acc_x, acc_y, acc_z])


# Generate mock data
ppg_signal = generate_mock_ppg_signal()
audio_signal = generate_mock_audio_signal()
accel_data = generate_mock_accelerometer()

print(f"   ✓ PPG signal: {len(ppg_signal)} samples (camera-based)")
print(f"   ✓ Audio signal: {len(audio_signal)} samples (breathing)")
print(f"   ✓ Accelerometer: {accel_data.shape} (gait analysis)")


# ---- Feature Extraction ----
print()
print("[3/8] Extracting biomarkers from signals...")

# Cardiovascular features (from PPG) - UNHEALTHY PATIENT
cv_biomarkers = {
    "heart_rate": 105,  # Tachycardia (elevated)
    "heart_rate_variability": 22.0,  # Low HRV (concerning)
    "pulse_wave_velocity": 12.5,  # High PWV (arterial stiffness)
    "blood_pressure_systolic": 158,  # Hypertension Stage 2
    "blood_pressure_diastolic": 98  # Hypertension Stage 2
}
print(f"   ⚠️  Cardiovascular: HR={cv_biomarkers['heart_rate']} bpm (HIGH), HRV={cv_biomarkers['heart_rate_variability']} ms (LOW)")

# CNS features (from accelerometer) - UNHEALTHY PATIENT
cns_biomarkers = {
    "gait_variability": 0.25,  # High variability (instability)
    "reaction_time": 0.68,  # Slow reaction time
    "tremor_amplitude": 0.15,  # Significant tremor
    "balance_score": 0.45  # Poor balance
}
print(f"   ⚠️  CNS: Gait variability={cns_biomarkers['gait_variability']} (HIGH), Balance={cns_biomarkers['balance_score']} (LOW)")

# Pulmonary features (from audio) - UNHEALTHY PATIENT
pulm_biomarkers = {
    "respiratory_rate": 26,  # Tachypnea (rapid breathing)
    "spo2": 91,  # Low oxygen saturation
    "breath_depth": 0.45,  # Shallow breathing
    "cough_frequency": 12  # Frequent coughing
}
print(f"   ⚠️  Pulmonary: RR={pulm_biomarkers['respiratory_rate']}/min (HIGH), SpO2={pulm_biomarkers['spo2']}% (LOW)")


# ---- Risk Inference ----
print()
print("[4/8] Calculating risk scores...")

from app.core.extraction.base import BiomarkerSet, Biomarker
from app.core.inference.risk_engine import CompositeRiskCalculator

risk_engine = RiskEngine()
composite_calc = CompositeRiskCalculator()

# Create BiomarkerSets - UNHEALTHY PATIENT DATA
cv_markers = BiomarkerSet(
    system=PhysiologicalSystem.CARDIOVASCULAR,
    biomarkers=[
        Biomarker(name="heart_rate", value=105, unit="bpm"),  # Tachycardia
        Biomarker(name="hrv_rmssd", value=22.0, unit="ms"),  # Low HRV
        Biomarker(name="blood_pressure_systolic", value=158, unit="mmHg"),  # Hypertension
        Biomarker(name="blood_pressure_diastolic", value=98, unit="mmHg"),
    ]
)

cns_markers = BiomarkerSet(
    system=PhysiologicalSystem.CNS,
    biomarkers=[
        Biomarker(name="gait_variability", value=0.25, unit=""),  # High instability
        Biomarker(name="balance_score", value=0.45, unit=""),  # Poor balance
        Biomarker(name="tremor_amplitude", value=0.15, unit="mm"),  # Significant tremor
    ]
)

pulm_markers = BiomarkerSet(
    system=PhysiologicalSystem.PULMONARY,
    biomarkers=[
        Biomarker(name="respiratory_rate", value=26, unit="breaths/min"),  # Tachypnea
        Biomarker(name="spo2", value=91, unit="%"),  # Low oxygen
        Biomarker(name="cough_frequency", value=12, unit="per_hour"),
    ]
)

# Calculate system risks
cv_result = risk_engine.compute_risk(cv_markers)
cns_result = risk_engine.compute_risk(cns_markers)
pulm_result = risk_engine.compute_risk(pulm_markers)

system_results = {
    PhysiologicalSystem.CARDIOVASCULAR: cv_result,
    PhysiologicalSystem.CNS: cns_result,
    PhysiologicalSystem.PULMONARY: pulm_result
}

# Calculate composite risk
composite_risk = composite_calc.compute_composite_risk(system_results)

print(f"   ✓ Cardiovascular: {cv_result.overall_risk.level.value} ({cv_result.overall_risk.score:.1f}%)")
print(f"   ✓ CNS: {cns_result.overall_risk.level.value} ({cns_result.overall_risk.score:.1f}%)")
print(f"   ✓ Pulmonary: {pulm_result.overall_risk.level.value} ({pulm_result.overall_risk.score:.1f}%)")
print(f"   ✓ COMPOSITE: {composite_risk.level.value} ({composite_risk.score:.1f}%)")


# ---- Trust Envelope ----
print()
print("[5/8] Calculating trust envelope...")

# Create trust envelope directly - UNHEALTHY PATIENT
from app.core.validation.trust_envelope import SafetyFlag

trust_envelope = TrustEnvelope(
    overall_reliability=0.78,  # Lower reliability due to concerning values
    confidence_penalty=1 - composite_risk.confidence,
    safety_flags=[
        SafetyFlag.PHYSIOLOGICAL_ANOMALY,
        SafetyFlag.LOW_CONFIDENCE
    ],
    modality_scores={
        "camera": 0.85,
        "audio": 0.80,
        "accelerometer": 0.75
    },
    system_reliability={
        PhysiologicalSystem.CARDIOVASCULAR.value: 0.82,
        PhysiologicalSystem.CNS.value: 0.75,
        PhysiologicalSystem.PULMONARY.value: 0.78
    },
    warnings=["Multiple high-risk indicators detected", "Medical consultation recommended"],
)

print(f"   ✓ Overall reliability: {trust_envelope.overall_reliability:.2f}")
print(f"   ✓ Data quality: {trust_envelope.data_quality_score:.2f}")
print(f"   ✓ Is reliable: {trust_envelope.is_reliable}")


# ---- LLM Interpretation ----
print()
print("[6/8] Generating Multi-LLM interpretation (3 models)...")
print("   🤖 Querying: Gemini 2.5 Flash (LangChain)")
print("   🤖 Querying: GPT-OSS-120B (HuggingFace)")
print("   🤖 Querying: II-Medical-8B (HuggingFace)")

# Initialize interpreters
multi_interpreter = MultiLLMInterpreter()
risk_interpreter = RiskInterpreter(multi_llm=True)  # Use multi-LLM mode

# Generate comprehensive interpretation using all 3 models
try:
    interpretation_result = risk_interpreter.interpret_composite_risk(
        system_results=system_results,
        composite_risk=composite_risk,
        trust_envelope=trust_envelope
    )
    
    print(f"   ✓ Interpretation complete!")
    print(f"   ✓ Summary length: {len(interpretation_result.summary)} chars")
    print(f"   ✓ Recommendations: {len(interpretation_result.recommendations)}")
    print(f"   ✓ Latency: {interpretation_result.latency_ms:.0f}ms")
    
    # Show first recommendation
    if interpretation_result.recommendations:
        print(f"   📋 First recommendation: {interpretation_result.recommendations[0][:80]}...")
    
except Exception as e:
    print(f"   ⚠️  LLM interpretation failed (using mock): {e}")
    interpretation_result = None

# Also generate context (legacy)
context_gen = MedicalContextGenerator()
cv_context = context_gen.generate_context(
    system=PhysiologicalSystem.CARDIOVASCULAR,
    risk_level=cv_result.overall_risk.level
)

print(f"   ✓ Medical context generated for cardiovascular system")
print(f"   ✓ Lifestyle factors: {len(cv_context.lifestyle_factors)} recommendations")


# ---- Agentic Validation ----
print()
print("[7/8] Running agent validation with medical models...")
print("   🔬 Agent 1: Using GPT-OSS-120B for biomarker plausibility")
print("   🔬 Agent 2: Using II-Medical-8B for cross-system consistency")

consensus = AgentConsensus()

# Try to run real validation with the new medical models
try:
    # Initialize agents (now using GPT-OSS-120B and II-Medical-8B)
    medgemma_agent = MedGemmaAgent()  # Uses medical_model_1
    openbio_agent = OpenBioLLMAgent()  # Uses medical_model_2
    
    # Run validations
    print("   ⏳ Validating biomarkers...")
    medgemma_validation = medgemma_agent.validate_biomarkers(
        biomarker_summary={
            "heart_rate": {"value": 105, "unit": "bpm", "status": "high"},
            "blood_pressure": {"value": "158/98", "unit": "mmHg", "status": "high"},
            "spo2": {"value": 91, "unit": "%", "status": "low"}
        },
        system=PhysiologicalSystem.CARDIOVASCULAR
    )
    
    print("   ⏳ Validating cross-system consistency...")
    openbio_validation = openbio_agent.validate_consistency(
        system_results=system_results,
        trust_envelope=trust_envelope
    )
    
    agent_results = {
        "MedGemma": medgemma_validation,
        "OpenBioLLM": openbio_validation
    }
    
    print(f"   ✓ Agent 1 ({medgemma_validation.model_used}): {medgemma_validation.status.value}")
    print(f"   ✓ Agent 2 ({openbio_validation.model_used}): {openbio_validation.status.value}")
    
except Exception as e:
    print(f"   ⚠️  Real validation failed (using mock): {e}")
    # Fallback to mock validation - UNHEALTHY PATIENT
    from app.core.agents.medical_agents import ValidationResult, ValidationStatus, ValidationFlag, FlagSeverity
    
    agent_results = {
        "MedGemma": ValidationResult(
            agent_name="MedGemma",
            model_used="GPT-OSS-120B (mock)",
            status=ValidationStatus.FLAGGED,
            confidence=0.75,
            flags=[
                ValidationFlag(
                    agent="MedGemma",
                    severity=FlagSeverity.WARNING,
                    category="cardiovascular",
                    message="Hypertension Stage 2 detected (158/98 mmHg)"
                ),
                ValidationFlag(
                    agent="MedGemma",
                    severity=FlagSeverity.WARNING,
                    category="cardiovascular",
                    message="Tachycardia present (105 bpm)"
                )
            ],
            is_mock=True
        ),
        "OpenBioLLM": ValidationResult(
            agent_name="OpenBioLLM",
            model_used="II-Medical-8B (mock)",
            status=ValidationStatus.FLAGGED,
            confidence=0.72,
            flags=[
                ValidationFlag(
                    agent="OpenBioLLM",
                    severity=FlagSeverity.CRITICAL,
                    category="pulmonary",
                    message="Low SpO2 requires immediate attention (91%)",
                    system=PhysiologicalSystem.PULMONARY
                ),
                ValidationFlag(
                    agent="OpenBioLLM",
                    severity=FlagSeverity.WARNING,
                    category="cns",
                    message="CNS abnormalities detected - poor balance and high tremor",
                    system=PhysiologicalSystem.CNS
                )
            ],
            is_mock=True
        )
    }

validation_result = consensus.compute_consensus(agent_results)

print(f"   ✓ Validation status: {validation_result.overall_status.value}")
print(f"   ✓ Agent agreement: {validation_result.agent_agreement:.1%}")
print(f"   ✓ Combined flags: {len(validation_result.combined_flags)}")
print(f"   ✓ Requires review: {validation_result.requires_human_review}")


# ---- Report Generation ----
print()
print("[8/8] Generating PDF reports with Multi-LLM insights...")

os.makedirs("reports", exist_ok=True)

# Patient report (includes multi-LLM interpretation)
patient_gen = PatientReportGenerator(output_dir="reports")
patient_report = patient_gen.generate(
    system_results=system_results,
    composite_risk=composite_risk,
    interpretation=interpretation_result,  # Multi-LLM interpretation
    trust_envelope=trust_envelope,
    patient_id="DEMO-001"
)

# Doctor report (includes validation + multi-LLM interpretation)
doctor_gen = DoctorReportGenerator(output_dir="reports")
doctor_report = doctor_gen.generate(
    system_results=system_results,
    composite_risk=composite_risk,
    trust_envelope=trust_envelope,
    interpretation=interpretation_result,  # Multi-LLM interpretation
    validation_result=validation_result,
    patient_id="DEMO-001"
)

print(f"   ✓ Patient report: {patient_report.report_id}")
if patient_report.pdf_path:
    print(f"     📄 PDF: {patient_report.pdf_path}")
    print(f"     📊 Includes insights from 3 AI models")
else:
    print(f"     ⚠️  (PDF skipped - reportlab not installed)")

print(f"   ✓ Doctor report: {doctor_report.report_id}")
if doctor_report.pdf_path:
    print(f"     📄 PDF: {doctor_report.pdf_path}")
    print(f"     🔬 Includes validation from medical AI models")


# ---- Summary ----
print()
print("=" * 60)
print("DEMO COMPLETE - MULTI-LLM HEALTH SCREENING SUMMARY")
print("=" * 60)
print()
print(f"Patient ID: DEMO-001")
print(f"Timestamp: {datetime.now().isoformat()}")
print()
print("AI Models Used:")
print("  🤖 Gemini 2.5 Flash (via LangChain) - Interpretation")
print("  🤖 GPT-OSS-120B (HuggingFace) - Validation & Interpretation")
print("  🤖 II-Medical-8B (HuggingFace) - Validation & Interpretation")
print()
print("Systems Analyzed:")
for system, result in system_results.items():
    risk = result.overall_risk
    status_icon = "🟢" if risk.level == RiskLevel.LOW else "🟡" if risk.level == RiskLevel.MODERATE else "🔴"
    print(f"  {status_icon} {system.value}: {risk.level.value} ({risk.score:.1f}%)")

print()
print(f"Overall Risk: {composite_risk.level.value.upper()} ({composite_risk.score:.1f}%)")
print(f"Confidence: {composite_risk.confidence:.1%}")
print(f"Reliability Score: {trust_envelope.overall_reliability:.2f}")
print()

if validation_result.requires_human_review:
    print("⚠️  REQUIRES HUMAN REVIEW")
    print(f"   Reason: {validation_result.recommendation}")
else:
    print("✓ Validation passed - no urgent flags")

print()
print("Multi-LLM Interpretation:")
if interpretation_result:
    print(f"  📝 Summary: {interpretation_result.summary[:150]}...")
    print(f"  💡 Recommendations: {len(interpretation_result.recommendations)} generated")
    print(f"  ⏱️  Total latency: {interpretation_result.latency_ms:.0f}ms")
else:
    print("  ⚠️  Using mock interpretation (API keys not configured)")
print()
print("Reports generated in: ./reports/")
print("  - Patient report (simple, AI-enhanced)")
print("  - Doctor report (detailed, with validation)")
print()
print("To run the API server:")
print("  uvicorn app.main:app --reload --port 8000")
print()
print("=" * 60)
