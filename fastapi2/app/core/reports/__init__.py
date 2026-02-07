"""
Report Generation Module

Generates downloadable PDF health screening reports.
Two types:
- Patient Report: Simple, color-coded, easy to understand
- Doctor Report: Detailed biomarkers, trust envelope, technical
"""
# Import V3 (Playwright-based) generators and export with V1-compatible names
from .patient_reportv3 import EnhancedPatientReportGeneratorV3 as PatientReportGenerator
from .patient_reportv3 import PatientReportV3 as PatientReport
from .doctor_reportv3 import DoctorReportGeneratorV3 as DoctorReportGenerator
from .doctor_reportv3 import DoctorReportV3 as DoctorReport

__all__ = [
    "PatientReportGenerator",
    "PatientReport",
    "DoctorReportGenerator",
    "DoctorReport",
]
