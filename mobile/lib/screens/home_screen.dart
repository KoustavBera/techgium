import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import '../providers/report_provider.dart';
import '../widgets/empty_state.dart';
import '../widgets/report_card.dart';
import '../widgets/delete_confirmation_dialog.dart';
import 'pdf_viewer_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: const _AppDrawer(),
      appBar: AppBar(
        title: const Text('Chiranjeevi'),
        centerTitle: false,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Settings coming soon')),
              );
            },
            tooltip: 'Settings',
          )
        ],
      ),
      body: const _HomeBody(),
      floatingActionButton: FloatingActionButton.large(
        heroTag: 'qr-scan-fab',
        onPressed: () {
          HapticFeedback.lightImpact();
          Navigator.pushNamed(context, '/scan');
        },
        tooltip: 'Scan QR Code',
        child: const Icon(Icons.qr_code_scanner_rounded, size: 36),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      bottomNavigationBar: BottomAppBar(
        shape: const CircularNotchedRectangle(),
        notchMargin: 10.0,
        height: 72,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            IconButton(
              icon: const Icon(Icons.home_rounded),
              onPressed: () {},
              tooltip: 'Home',
              color: Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(width: 72),
            IconButton(
              icon: const Icon(Icons.history_rounded),
              onPressed: () => Navigator.pushNamed(context, '/history'),
              tooltip: 'History',
            ),
          ],
        ),
      ),
    );
  }
}

class _AppDrawer extends StatelessWidget {
  const _AppDrawer();

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return NavigationDrawer(
      onDestinationSelected: (index) {
        // Handle navigation and close drawer (handled automatically by pushing route)
        Navigator.pop(context); // Close drawer first
        if (index == 0) {
          Navigator.pushNamed(context, '/scan');
        } else if (index == 1) {
          Navigator.pushNamed(context, '/history');
        }
      },
      children: [
        DrawerHeader(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                colorScheme.primary,
                colorScheme.secondary,
              ],
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.health_and_safety_rounded, size: 48, color: colorScheme.onPrimary),
              const SizedBox(height: 12),
              Text(
                'Chiranjeevi',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      color: colorScheme.onPrimary,
                      fontWeight: FontWeight.w800,
                    ),
              ),
              Text(
                'Health Report Companion',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: colorScheme.onPrimary.withValues(alpha: 0.8),
                    ),
              ),
            ],
          ),
        ),
        NavigationDrawerDestination(
          icon: const Icon(Icons.qr_code_scanner_rounded),
          label: const Text('Scan QR Code'),
        ),
        NavigationDrawerDestination(
          icon: const Icon(Icons.history_rounded),
          label: const Text('Report History'),
        ),
        const Divider(),
        const ListTile(
          leading: Icon(Icons.settings_outlined),
          title: Text('Settings'),
          enabled: false,
          subtitle: Text('Coming soon'),
        ),
        ListTile(
          leading: const Icon(Icons.info_outline_rounded),
          title: const Text('About'),
          onTap: () {
            showAboutDialog(
              context: context,
              applicationName: 'Chiranjeevi',
              applicationVersion: '1.0.0',
              applicationIcon: Icon(Icons.health_and_safety_rounded, color: colorScheme.primary, size: 48),
              applicationLegalese: '© 2026 Techgium',
            );
          },
        ),
      ],
    );
  }
}

class _HomeBody extends StatelessWidget {
  const _HomeBody();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16.0),
      children: [
        const _HeroBannerCard(),
        const SizedBox(height: 32),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('Recent Reports',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700)),
            TextButton(
              onPressed: () => Navigator.pushNamed(context, '/history'),
              child: const Text('See All'),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Consumer<ReportProvider>(
          builder: (context, provider, _) {
            if (provider.isLoading) {
              return const Center(child: CircularProgressIndicator());
            }
            if (provider.reports.isEmpty) {
              return const EmptyState(
                icon: Icons.folder_open_rounded,
                title: 'No Recent Reports',
                subtitle: 'Reports you download will appear here',
              );
            }
            // Show up to 3 most recent
            final recent = provider.reports.take(3).toList();
            return Column(
              children: recent.asMap().entries.map((entry) => ReportCard(
                report: entry.value,
                animationDelay: Duration(milliseconds: entry.key * 80),
                onView: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => PdfViewerScreen(report: entry.value),
                  ),
                ),
                onDelete: () => showDialog(
                  context: context,
                  builder: (_) => DeleteConfirmationDialog(
                    reportId: entry.value.reportId,
                    onConfirm: () =>
                        Provider.of<ReportProvider>(context, listen: false)
                            .deleteReport(entry.value),
                  ),
                ),
              )).toList(),
            );
          },
        ),
      ],
    );
  }
}

class _HeroBannerCard extends StatelessWidget {
  const _HeroBannerCard();

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Card(
      child: Container(
        padding: const EdgeInsets.all(24.0),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          gradient: LinearGradient(
            colors: [
              colorScheme.primaryContainer,
              colorScheme.secondaryContainer.withValues(alpha: 0.5),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.8, end: 1.0),
                  duration: const Duration(milliseconds: 1000),
                  curve: Curves.elasticOut,
                  builder: (context, value, child) => Transform.scale(
                    scale: value,
                    child: child,
                  ),
                  child: Icon(
                    Icons.qr_code_scanner_rounded,
                    size: 48,
                    color: colorScheme.onPrimaryContainer,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Text(
                    'Ready to scan',
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          color: colorScheme.onPrimaryContainer,
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              'Point camera at QR code to get patient report',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: colorScheme.onPrimaryContainer.withValues(alpha: 0.8),
                  ),
            ),
          ],
        ),
      ),
    );
  }
}
