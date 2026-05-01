import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../services/pdf_service.dart';

class DownloadProgressSheet extends StatefulWidget {
  final String reportId;
  final String url;
  final void Function(String filePath, int fileSizeBytes) onComplete;
  final VoidCallback onCancel;
  final void Function(String errorMessage) onError;

  const DownloadProgressSheet({
    super.key,
    required this.reportId,
    required this.url,
    required this.onComplete,
    required this.onCancel,
    required this.onError,
  });

  @override
  State<DownloadProgressSheet> createState() => _DownloadProgressSheetState();
}

class _DownloadProgressSheetState extends State<DownloadProgressSheet> {
  double _progress = 0.0;
  int _receivedBytes = 0;
  int _totalBytes = 0;
  bool _isCancelled = false;
  late final CancelToken _cancelToken;
  late final PdfService _pdfService;

  @override
  void initState() {
    super.initState();
    _cancelToken = CancelToken();
    _pdfService = PdfService();
    _startDownload();
  }

  Future<void> _startDownload() async {
    try {
      final filePath = await _pdfService.downloadReport(
        url: widget.url,
        reportId: widget.reportId,
        cancelToken: _cancelToken,
        onProgress: (received, total) {
          if (!mounted || _isCancelled) return;
          setState(() {
            _receivedBytes = received;
            _totalBytes = total;
            _progress = total > 0 ? received / total : 0;
          });
        },
      );

      final file = File(filePath);
      final fileSize = await file.length();

      if (mounted && !_isCancelled) {
        HapticFeedback.heavyImpact();
        widget.onComplete(filePath, fileSize);
      }
    } on DioException catch (e) {
      if (e.type == DioExceptionType.cancel) return;
      if (mounted) widget.onError(_userFriendlyError(e));
    } catch (e) {
      if (mounted) widget.onError('Unexpected error: $e');
    }
  }

  String _userFriendlyError(DioException e) {
    return switch (e.type) {
      DioExceptionType.connectionTimeout =>
        'Connection timed out. Make sure you\'re on the same Wi-Fi as the screening device.',
      DioExceptionType.connectionError =>
        'Network error. Make sure you\'re on the same Wi-Fi as the screening device.',
      DioExceptionType.badResponse =>
        'Server error (${e.response?.statusCode}). The report may have been deleted.',
      _ => 'Download failed. Please try again.',
    };
  }

  String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    return PopScope(
      canPop: false,
      child: Container(
        padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
        decoration: BoxDecoration(
          color: colorScheme.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 32,
                  height: 4,
                  decoration: BoxDecoration(
                    color: colorScheme.onSurfaceVariant.withValues(alpha: 0.4),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 24),
              Row(
                children: [
                  Icon(Icons.download_rounded, color: colorScheme.primary, size: 28),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Downloading Report', style: textTheme.titleLarge),
                        Text(
                          'ID: ${widget.reportId}',
                          style: textTheme.bodyMedium?.copyWith(
                            color: colorScheme.onSurfaceVariant,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 32),
              LinearProgressIndicator(
                value: _progress > 0 ? _progress : null,
                borderRadius: BorderRadius.circular(4),
                minHeight: 8,
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    _totalBytes > 0
                        ? '${_formatBytes(_receivedBytes)} / ${_formatBytes(_totalBytes)}'
                        : _formatBytes(_receivedBytes),
                    style: textTheme.bodySmall?.copyWith(
                      color: colorScheme.onSurfaceVariant,
                    ),
                  ),
                  Text(
                    '${(_progress * 100).toInt()}%',
                    style: textTheme.labelLarge?.copyWith(
                      color: colorScheme.primary,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 32),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton(
                  onPressed: () {
                    HapticFeedback.lightImpact();
                    _isCancelled = true;
                    _cancelToken.cancel();
                    widget.onCancel();
                    Navigator.pop(context);
                  },
                  child: const Text('Cancel Download'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
