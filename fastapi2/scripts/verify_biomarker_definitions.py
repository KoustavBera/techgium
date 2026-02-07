"""
Quick verification script to check biomarker definitions coverage.
"""
import sys
sys.path.append('.')

from app.core.reports.patient_reportv3 import BIOMARKER_DEFINITIONS, BIOMARKER_NAMES

# Collect all biomarkers from the codebase
known_biomarkers = set([
    # Cardiovascular
    "heart_rate", "hrv_rmssd", "systolic_bp", "diastolic_bp", 
    "thoracic_impedance", "chest_micro_motion",
    
    # Pulmonary
    "spo2", "respiratory_rate", "respiration_rate", 
    "breath_depth", "breathing_depth", "breath_depth_index",
    
    # Skin/Thermal
    "surface_temperature_avg", "skin_temperature", "skin_temperature_max",
    "thermal_asymmetry", "nostril_thermal_asymmetry",
    "texture_roughness", "skin_redness", "skin_yellowness",
    "color_uniformity", "lesion_count",
    
    # CNS
    "gait_variability", "balance_score", "tremor_resting", "tremor_postural",
    "tremor_kinetic", "tremor_intentional", "reaction_time", 
    "posture_entropy", "cns_stability_score", 
    "sway_amplitude_ap", "sway_amplitude_ml",
    
    # Skeletal
    "posture_score", "symmetry_index", "gait_symmetry_ratio",
    "step_length_symmetry", "stance_stability_score",
    "sway_velocity", "average_joint_rom",
    
    # Nasal
    "nostril_occlusion_score", "respiratory_effort_index", "nasal_cycle_balance",
    
    # GI
    "abdominal_rhythm_score", "visceral_motion_variance", "abdominal_respiratory_rate",
    
    # Renal
    "fluid_asymmetry_index", "total_body_water_proxy",
    "extracellular_fluid_ratio", "fluid_overload_index",
    
    # Reproductive/Hormonal
    "autonomic_imbalance_index", "stress_response_proxy",
    "regional_flow_variability", "thermoregulation_proxy",
])

# Check coverage
missing_definitions = []
for biomarker in known_biomarkers:
    if biomarker not in BIOMARKER_DEFINITIONS:
        missing_definitions.append(biomarker)

print(f"Total known biomarkers: {len(known_biomarkers)}")
print(f"Definitions available: {len(BIOMARKER_DEFINITIONS)}")
print(f"Missing definitions: {len(missing_definitions)}")

if missing_definitions:
    print("\n⚠ Missing Definitions:")
    for m in sorted(missing_definitions):
        print(f"  - {m}")
else:
    print("\n✅ All biomarkers have definitions!")

# Check friendly names
missing_names = []
for biomarker in known_biomarkers:
    if biomarker not in BIOMARKER_NAMES:
        missing_names.append(biomarker)

print(f"\nFriendly names available: {len(BIOMARKER_NAMES)}")
print(f"Missing names: {len(missing_names)}")

if missing_names:
    print("\n⚠ Missing Friendly Names:")
    for m in sorted(missing_names):
        print(f"  - {m}")
else:
    print("\n✅ All biomarkers have friendly names!")
