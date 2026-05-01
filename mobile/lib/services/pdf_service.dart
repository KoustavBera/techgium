import 'dart:io';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';

class PdfService {
  final Dio _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 10),
    receiveTimeout: const Duration(minutes: 5),
    headers: {'Accept': 'application/pdf'},
  ));

  Future<String> downloadReport({
    required String url,
    required String reportId,
    required void Function(int received, int total) onProgress,
    CancelToken? cancelToken,
  }) async {
    final dir = await getApplicationDocumentsDirectory();
    final reportsDir = Directory('${dir.path}/reports');
    if (!await reportsDir.exists()) await reportsDir.create(recursive: true);
    final savePath = '${reportsDir.path}/$reportId.pdf';

    await _dio.download(
      url,
      savePath,
      onReceiveProgress: onProgress,
      cancelToken: cancelToken,
    );

    return savePath;
  }

  Future<void> deleteReport(String filePath) async {
    final file = File(filePath);
    if (await file.exists()) await file.delete();
  }

  Future<bool> reportFileExists(String reportId) async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/reports/$reportId.pdf').exists();
  }

  Future<File?> getReportFile(String reportId) async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/reports/$reportId.pdf');
    return await file.exists() ? file : null;
  }

  static String? extractReportId(String url) {
    // Matches: http://<host>/api/v1/reports/<report_id>/download
    final uri = Uri.tryParse(url);
    if (uri == null) return null;
    final segments = uri.pathSegments;
    // Expected: ['api', 'v1', 'reports', '<id>', 'download']
    if (segments.length == 5 &&
        segments[0] == 'api' &&
        segments[1] == 'v1' &&
        segments[2] == 'reports' &&
        segments[4] == 'download') {
      return segments[3];
    }
    return null;
  }
}
