import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/report_metadata.dart';

class ReportCard extends StatefulWidget {
  final ReportMetadata report;
  final VoidCallback onView;
  final VoidCallback onDelete;
  final Duration animationDelay;

  const ReportCard({
    super.key,
    required this.report,
    required this.onView,
    required this.onDelete,
    this.animationDelay = Duration.zero,
  });

  @override
  State<ReportCard> createState() => _ReportCardState();
}

class _ReportCardState extends State<ReportCard>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<Offset> _slideAnim;
  late final Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 450),
    );
    _slideAnim = Tween<Offset>(
      begin: const Offset(0, 0.25),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic));
    _fadeAnim = Tween<double>(begin: 0.0, end: 1.0)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeIn));

    Future.delayed(widget.animationDelay, () {
      if (mounted) _ctrl.forward();
    });
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  String _formatBytes(int? bytes) {
    if (bytes == null) return 'Unknown size';
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;
    final formattedDate = DateFormat.yMMMEd().format(widget.report.downloadedAt);

    return SlideTransition(
      position: _slideAnim,
      child: FadeTransition(
        opacity: _fadeAnim,
        child: Card(
          margin: const EdgeInsets.only(bottom: 12.0),
          child: InkWell(
            onTap: widget.onView,
        borderRadius: BorderRadius.circular(20),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            children: [
              // File icon
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: colorScheme.primaryContainer,
                  shape: BoxShape.circle,
                ),
                child: Icon(Icons.picture_as_pdf_rounded,
                    color: colorScheme.onPrimaryContainer, size: 24),
              ),
              const SizedBox(width: 16),
              // Report details
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.report.reportId,
                      style: textTheme.titleMedium,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      formattedDate,
                      style: textTheme.bodySmall?.copyWith(
                          color: colorScheme.onSurfaceVariant),
                    ),
                    Text(
                      _formatBytes(widget.report.fileSizeBytes),
                      style: textTheme.bodySmall?.copyWith(
                          color: colorScheme.onSurfaceVariant),
                    ),
                  ],
                ),
              ),
              // Actions
              Column(
                children: [
                  FilledButton.tonal(
                    onPressed: widget.onView,
                    style: FilledButton.styleFrom(
                      minimumSize: const Size(60, 36),
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                    ),
                    child: const Text('View'),
                  ),
                  const SizedBox(height: 4),
                  IconButton(
                    icon: Icon(Icons.delete_outline_rounded,
                        color: colorScheme.error),
                    onPressed: widget.onDelete,
                    tooltip: 'Delete report',
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    ),
    ),
    );
  }
}
