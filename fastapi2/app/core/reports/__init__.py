"""
Report Generation Module

Generates downloadable PDF health screening reports.
Two types:
- Patient Report: Simple, color-coded, easy to understand
- Doctor Report: Detailed biomarkers, trust envelope, technical
"""
# Import standard (ReportLab-based) generators
from .patient_report import EnhancedPatientReportGenerator as PatientReportGenerator
from .patient_report import PatientReport
from .doctor_report import DoctorReportGenerator
from .doctor_report import DoctorReport

__all__ = [
    "PatientReportGenerator",
    "PatientReport",
    "DoctorReportGenerator",
    "DoctorReport",
]
