"""
Report Generation Module

Generates downloadable PDF health screening reports.
Two types:
- Patient Report: Simple, color-coded, easy to understand
- Doctor Report: Detailed biomarkers, trust envelope, technical
"""
# Import from optimised generator
from .patient_report_optimised import EnhancedPatientReportGenerator as PatientReportGenerator
from .patient_report_optimised import PatientReport
from .doctor_report import DoctorReportGenerator, DoctorReport

__all__ = [
    "PatientReportGenerator",
    "PatientReport",
    "DoctorReportGenerator",
    "DoctorReport",
]
