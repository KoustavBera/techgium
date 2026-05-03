import 'package:animations/animations.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/report_provider.dart';
import '../widgets/empty_state.dart';
import '../widgets/report_card.dart';
import '../widgets/delete_confirmation_dialog.dart';
import 'pdf_viewer_screen.dart';

PageRouteBuilder<T> _sharedAxisRoute<T>(Widget page) {
  return PageRouteBuilder<T>(
    pageBuilder: (_, __, ___) => page,
    transitionDuration: const Duration(milliseconds: 350),
    reverseTransitionDuration: const Duration(milliseconds: 300),
    transitionsBuilder: (_, animation, secondaryAnimation, child) {
      return SharedAxisTransition(
        animation: animation,
        secondaryAnimation: secondaryAnimation,
        transitionType: SharedAxisTransitionType.scaled,
        child: child,
      );
    },
  );
}

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Report History'),
        actions: [
          Consumer<ReportProvider>(
            builder: (_, provider, __) => provider.count > 0
                ? Padding(
                    padding: const EdgeInsets.only(right: 16),
                    child: Chip(
                      label: Text('${provider.count} report${provider.count != 1 ? 's' : ''}'),
                      avatar: const Icon(Icons.folder_rounded, size: 16),
                    ),
                  )
                : const SizedBox.shrink(),
          ),
        ],
      ),
      body: Consumer<ReportProvider>(
        builder: (context, provider, _) {
          if (provider.isLoading) {
            return const Center(child: CircularProgressIndicator());
          }
          if (provider.reports.isEmpty) {
            return EmptyState(
              icon: Icons.folder_open_rounded,
              title: 'No Reports Yet',
              subtitle: 'Scan a QR code to download\nyour first health report',
              actionLabel: 'Scan Now',
              actionOnPressed: () => Navigator.pushNamed(context, '/scan'),
            );
          }
          return RefreshIndicator(
            onRefresh: provider.loadReports,
            child: ListView.builder(
              padding: const EdgeInsets.all(16.0),
              itemCount: provider.reports.length,
              itemBuilder: (context, index) {
                final report = provider.reports[index];
                return ReportCard(
                  report: report,
                  animationDelay: Duration(milliseconds: index * 80),
                  onView: () => Navigator.push(
                    context,
                    _sharedAxisRoute(PdfViewerScreen(report: report)),
                  ),
                  onDelete: () => showDialog(
                    context: context,
                    builder: (_) => DeleteConfirmationDialog(
                      reportId: report.reportId,
                      onConfirm: () => provider.deleteReport(report),
                    ),
                  ),
                );
              },
            ),
          );
        },
      ),
    );
  }
}
