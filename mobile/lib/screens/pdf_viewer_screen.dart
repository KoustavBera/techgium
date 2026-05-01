import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:pdfrx/pdfrx.dart';
import 'package:share_plus/share_plus.dart';
import '../models/report_metadata.dart';

class PdfViewerScreen extends StatelessWidget {
  final ReportMetadata report;

  const PdfViewerScreen({super.key, required this.report});

  Future<void> _shareReport() async {
    final xFile = XFile(report.filePath);
    await SharePlus.instance.share(
      ShareParams(
        text: 'Chiranjeevi Health Report',
        files: [xFile],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final formattedDate = DateFormat.yMMMd().format(report.downloadedAt);

    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Report',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            Text(
              formattedDate,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_rounded),
            onPressed: _shareReport,
            tooltip: 'Share PDF',
          ),
        ],
      ),
      body: PdfViewer.file(
        report.filePath,
      ),
    );
  }
}
