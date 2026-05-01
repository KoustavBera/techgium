class ReportMetadata {
  final int? id;
  final String reportId;
  final String? patientName;
  final DateTime downloadedAt;
  final String filePath;
  final int? fileSizeBytes;
  final String? sourceUrl;

  const ReportMetadata({
    this.id,
    required this.reportId,
    this.patientName,
    required this.downloadedAt,
    required this.filePath,
    this.fileSizeBytes,
    this.sourceUrl,
  });

  Map<String, dynamic> toMap() => {
    'report_id': reportId,
    'patient_name': patientName,
    'downloaded_at': downloadedAt.toIso8601String(),
    'file_path': filePath,
    'file_size_bytes': fileSizeBytes,
    'source_url': sourceUrl,
  };

  factory ReportMetadata.fromMap(Map<String, dynamic> map) => ReportMetadata(
    id: map['id'] as int?,
    reportId: map['report_id'] as String,
    patientName: map['patient_name'] as String?,
    downloadedAt: DateTime.parse(map['downloaded_at'] as String),
    filePath: map['file_path'] as String,
    fileSizeBytes: map['file_size_bytes'] as int?,
    sourceUrl: map['source_url'] as String?,
  );
}
