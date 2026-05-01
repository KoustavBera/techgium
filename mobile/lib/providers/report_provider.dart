import 'package:flutter/material.dart';
import '../models/report_metadata.dart';
import '../services/db_service.dart';
import '../services/pdf_service.dart';

class ReportProvider extends ChangeNotifier {
  final DatabaseService _db = DatabaseService();
  List<ReportMetadata> _reports = [];
  bool _isLoading = false;

  List<ReportMetadata> get reports => _reports;
  bool get isLoading => _isLoading;
  int get count => _reports.length;

  Future<void> loadReports() async {
    _isLoading = true;
    notifyListeners();
    _reports = await _db.getAllReports();
    _isLoading = false;
    notifyListeners();
  }

  Future<void> deleteReport(ReportMetadata report) async {
    await PdfService().deleteReport(report.filePath);
    await _db.deleteReport(report.reportId);
    await loadReports();
  }
}
