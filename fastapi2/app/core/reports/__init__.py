"""
Report Generation Module

Generates downloadable PDF health screening reports.
Two types:
- Patient Report: Simple, color-coded, easy to understand
- Doctor Report: Detailed biomarkers, trust envelope, technical
"""
from .patient_report import PatientReportGenerator, PatientReport
from .doctor_report import DoctorReportGenerator, DoctorReport

__all__ = [
    "PatientReportGenerator",
    "PatientReport",
    "DoctorReportGenerator",
    "DoctorReport",
]
