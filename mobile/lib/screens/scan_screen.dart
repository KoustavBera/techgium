import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import '../services/db_service.dart';
import '../services/pdf_service.dart';
import '../models/report_metadata.dart';
import '../widgets/scan_overlay.dart';
import '../widgets/download_progress_sheet.dart';
import '../providers/report_provider.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  late final MobileScannerController _controller;
  bool _isProcessing = false;
  bool _torchEnabled = false;
  bool _showSuccessFlash = false;

  @override
  void initState() {
    super.initState();
    _controller = MobileScannerController(
      detectionSpeed: DetectionSpeed.noDuplicates,
      facing: CameraFacing.back,
      torchEnabled: false,
    );
    WidgetsBinding.instance.addPostFrameCallback((_) => _checkCameraPermission());
  }

  Future<void> _checkCameraPermission() async {
    final status = await Permission.camera.status;
    if (status.isGranted) return;

    if (status.isPermanentlyDenied) {
      _showPermissionDeniedDialog(permanent: true);
      return;
    }

    final result = await Permission.camera.request();
    if (!result.isGranted && mounted) {
      _showPermissionDeniedDialog(permanent: result.isPermanentlyDenied);
    }
  }

  void _showPermissionDeniedDialog({required bool permanent}) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        icon: const Icon(Icons.camera_alt_outlined),
        title: const Text('Camera Access Required'),
        content: Text(
          permanent
            ? 'Camera access was permanently denied. Please enable it in your device settings to scan QR codes.'
            : 'Chiranjeevi needs camera access to scan health report QR codes.',
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              Navigator.pop(context);
            },
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.pop(ctx);
              if (permanent) {
                openAppSettings();
              } else {
                _checkCameraPermission();
              }
            },
            child: Text(permanent ? 'Open Settings' : 'Grant Access'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onDetect(BarcodeCapture capture) async {
    if (_isProcessing) return;

    final rawValue = capture.barcodes.firstOrNull?.rawValue;
    if (rawValue == null) return;

    final reportId = PdfService.extractReportId(rawValue);
    if (reportId == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text('Not a valid Chiranjeevi health report QR code'),
            backgroundColor: Theme.of(context).colorScheme.error,
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
      return;
    }

    HapticFeedback.mediumImpact();
    setState(() {
      _isProcessing = true;
      _showSuccessFlash = true;
    });
    _controller.stop();

    await Future.delayed(const Duration(milliseconds: 600));
    if (mounted) {
      setState(() => _showSuccessFlash = false);
    }

    final exists = await DatabaseService().reportExists(reportId);
    if (exists && mounted) {
      _showAlreadyDownloadedDialog(reportId, rawValue);
      return;
    }

    if (mounted) {
      _showDownloadConfirmation(reportId, rawValue);
    }
  }

  void _showAlreadyDownloadedDialog(String reportId, String url) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        icon: const Icon(Icons.info_outline_rounded),
        title: const Text('Report Already Downloaded'),
        content: Text('You have already downloaded report $reportId. Do you want to download it again?'),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              setState(() => _isProcessing = false);
              _controller.start();
            },
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.pop(context);
              _startDownload(reportId, url);
            },
            child: const Text('Redownload'),
          ),
        ],
      ),
    );
  }

  void _showDownloadConfirmation(String reportId, String url) {
    showModalBottomSheet(
      context: context,
      isDismissible: false,
      builder: (sheetContext) => PopScope(
        canPop: false,
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.check_circle_outline, color: Theme.of(sheetContext).colorScheme.primary, size: 28),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Text('Report Found', style: Theme.of(sheetContext).textTheme.titleLarge),
                    ),
                  ],
                ),
                const SizedBox(height: 24),
                Text('ID: $reportId', style: Theme.of(sheetContext).textTheme.bodyLarge),
                const SizedBox(height: 8),
                Text(
                  'Source: ${Uri.parse(url).host}',
                  style: Theme.of(sheetContext).textTheme.bodyMedium?.copyWith(
                    color: Theme.of(sheetContext).colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 32),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    TextButton(
                      onPressed: () {
                        Navigator.pop(context);
                        setState(() => _isProcessing = false);
                        _controller.start();
                      },
                      child: const Text('Cancel'),
                    ),
                    const SizedBox(width: 12),
                    FilledButton.icon(
                      onPressed: () {
                        Navigator.pop(context);
                        _startDownload(reportId, url);
                      },
                      icon: const Icon(Icons.download_rounded),
                      label: const Text('Download Report'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _startDownload(String reportId, String url) {
    showModalBottomSheet(
      context: context,
      isDismissible: false,
      isScrollControlled: true,
      builder: (_) => DownloadProgressSheet(
        reportId: reportId,
        url: url,
        onComplete: (filePath, fileSize) async {
          await DatabaseService().insertReport(ReportMetadata(
            reportId: reportId,
            downloadedAt: DateTime.now(),
            filePath: filePath,
            fileSizeBytes: fileSize,
            sourceUrl: url,
          ));
          if (mounted) {
            // Trigger provider update
            Provider.of<ReportProvider>(context, listen: false).loadReports();
            Navigator.pop(context); // Close progress sheet
            Navigator.pop(context); // Back to home
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: const Text('Report downloaded successfully!'),
                behavior: SnackBarBehavior.floating,
                action: SnackBarAction(
                  label: 'View History',
                  onPressed: () => Navigator.pushNamed(context, '/history'),
                ),
              ),
            );
          }
        },
        onCancel: () {
          setState(() => _isProcessing = false);
          _controller.start();
        },
        onError: (message) {
          Navigator.pop(context);
          setState(() => _isProcessing = false);
          _controller.start();
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(message), backgroundColor: Theme.of(context).colorScheme.error),
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: Colors.white,
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset(
              'assets/images/heart_icon.png',
              height: 24,
              width: 24,
              fit: BoxFit.contain,
            ),
            const SizedBox(width: 8),
            const Text('Scan QR Code'),
          ],
        ),
        actions: [
          IconButton(
            icon: Icon(_torchEnabled ? Icons.flash_on : Icons.flash_off),
            onPressed: () {
              HapticFeedback.selectionClick();
              _controller.toggleTorch();
              setState(() => _torchEnabled = !_torchEnabled);
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          MobileScanner(
            controller: _controller,
            onDetect: _onDetect,
          ),
          ScanOverlay(isScanning: _isProcessing),
          if (_showSuccessFlash) const _ScanSuccessFlash(),
        ],
      ),
    );
  }
}

class _ScanSuccessFlash extends StatefulWidget {
  const _ScanSuccessFlash();
  @override
  State<_ScanSuccessFlash> createState() => _ScanSuccessFlashState();
}

class _ScanSuccessFlashState extends State<_ScanSuccessFlash>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _scaleAnim;
  late final Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 500))
      ..forward();
    _scaleAnim = Tween(begin: 0.5, end: 1.0)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.elasticOut));
    _fadeAnim = Tween(begin: 0.0, end: 1.0)
        .animate(CurvedAnimation(parent: _ctrl,
            curve: const Interval(0.0, 0.4, curve: Curves.easeIn)));
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ScaleTransition(
        scale: _scaleAnim,
        child: FadeTransition(
          opacity: _fadeAnim,
          child: Container(
            width: 120,
            height: 120,
            decoration: const BoxDecoration(
              color: Color(0xFF4CAF50),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.check_rounded,
                color: Colors.white, size: 64),
          ),
        ),
      ),
    );
  }
}
