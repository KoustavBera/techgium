import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class DeleteConfirmationDialog extends StatelessWidget {
  final String reportId;
  final VoidCallback onConfirm;

  const DeleteConfirmationDialog({
    super.key,
    required this.reportId,
    required this.onConfirm,
  });

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return AlertDialog(
      icon: Icon(Icons.delete_forever_rounded,
          color: colorScheme.error, size: 32),
      title: const Text('Delete Report?'),
      content: Text(
        'This will permanently delete report "$reportId" from your device. This cannot be undone.',
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          style: FilledButton.styleFrom(
            backgroundColor: colorScheme.error,
            foregroundColor: colorScheme.onError,
          ),
          onPressed: () {
            HapticFeedback.heavyImpact();
            Navigator.pop(context);
            onConfirm();
          },
          child: const Text('Delete'),
        ),
      ],
    );
  }
}
